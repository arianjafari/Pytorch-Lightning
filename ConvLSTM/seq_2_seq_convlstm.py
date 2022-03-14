# This is written by Arian Jafari, but adopted from https://github.com/aserdega
# Email: arian.jafari@gmail.com


import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTM
from cnn import ConvEncoder, ConvDecoder

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self,
                hidden_dim = 64,
                batch_size = 32,
                hidden_spt = 16,
                lstm_dims = [64, 64],
                img_size = 64,
                img_channel = 5,
                out_channel = 1,
                print_step = 500,
                alpha = 4000,
                beta = 16000,
                norm = nn.InstanceNorm2d,
                teacher_forcing = True,
                ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_spt = hidden_spt
        self.lstm_dims = lstm_dims
        self.img_size = img_size
        self.img_channel = img_channel
        self.out_channel = out_channel
        self.print_step = print_step
        self.alpha = alpha
        self.beta = beta
        self.norm = norm
        self.teacher_forcing = teacher_forcing

        # cnn_encoder_input to encode the input which has 5 channel
        self.cnn_encoder_input = ConvEncoder(in_channel = self.img_channel, out_dim = self.hidden_dim, norm = self.norm)
        # cnn_encoder_label to encode the label which has 1 channel
        self.cnn_encoder_label = ConvEncoder(in_channel = self.out_channel, out_dim = self.hidden_dim, norm = self.norm)
        
        self.cnn_decoder = ConvDecoder(b_size = self.batch_size,
                                       out_channel = self.out_channel,
                                       inp_dim = self.hidden_dim,
                                       norm = self.norm)
        
        self.lstm_encoder = ConvLSTM(
                   input_size = (self.hidden_spt,self.hidden_spt),
                   input_dim = self.hidden_dim,
                   hidden_dim = self.lstm_dims,
                   kernel_size = (3,3),
                   num_layers = len(self.lstm_dims),
                   peephole = True,
                   batchnorm = False,
                   batch_first = True,
                   activation = torch.tanh
                  )

        self.lstm_decoder = ConvLSTM(
                   input_size = (self.hidden_spt,self.hidden_spt),
                   input_dim = self.hidden_dim,
                   hidden_dim = self.lstm_dims,
                   kernel_size = (3,3),
                   num_layers = len(self.lstm_dims),
                   peephole = True,
                   batchnorm = False,
                   batch_first = True,
                   activation = torch.tanh
                  )

    def get_sample_prob(self, step):
        
        return self.alpha / (self.alpha + np.exp((step + self.beta) / self.alpha))
        
    def autoencoder(self, x, y, x_seq_len, y_seq_len, step , is_train = True):
    
        nextf_raw = y # the grand truth frames

        #----cnn encoder----
        # stack each sequence of B,T,C,H,W as BT,C,H,W to be fed in CNN model
        # prevf_raw = x[:,:x_seq_len,:,:,:].contiguous().view(-1,self.img_channel,self.img_size,self.img_size)
        prevf_raw = x.contiguous().view(-1,self.img_channel,self.img_size,self.img_size)
        prevf_enc = self.cnn_encoder_input(prevf_raw).view(
                                                    self.batch_size,
                                                    x_seq_len,
                                                    self.hidden_dim,
                                                    self.hidden_spt,self.hidden_spt)
        # print("prevf_raw shape: ", prevf_raw.shape)
        # print("prevf_enc shape: ", prevf_enc.shape)

        if (is_train and self.teacher_forcing):
            cnn_encoder_out = self.cnn_encoder_label(nextf_raw.contiguous().view(-1,
                                                                           self.out_channel,
                                                                           self.img_size,
                                                                           self.img_size))
            nextf_enc       = cnn_encoder_out.view(
                                                   self.batch_size,
                                                   y_seq_len,
                                                   self.hidden_dim,
                                                   self.hidden_spt,self.hidden_spt)

            # print("nextf_enc shape: ", nextf_enc.shape)

        #----lstm encoder---

        hidden           = self.lstm_encoder.get_init_states(self.batch_size)
        _, encoder_state = self.lstm_encoder(prevf_enc, hidden)

        #----lstm decoder---

        # sample_prob =  self.get_sample_prob(step) if (is_train and self.teacher_forcing) else 0
        sample_prob =  0.5 if (is_train and self.teacher_forcing) else 0
        decoder_output_list = []
        r_hist = []

        for s in range(y_seq_len):
            if s == 0:
                decoder_out, decoder_state = self.lstm_decoder(prevf_enc[:,-1:,:,:,:], encoder_state)
            else:
                r = np.random.rand()
                r_hist.append(int(r > sample_prob)) #debug

                if r > sample_prob:
                    decoder_out, decoder_state = self.lstm_decoder(decoder_out, decoder_state)
                else:
                    decoder_out, decoder_state = self.lstm_decoder(nextf_enc[:,s-1:s,:,:,:], decoder_state)
            
            decoder_output_list.append(decoder_out)
        
        final_decoder_out = torch.cat(decoder_output_list, 1)
        # print("final_decoder_out shape: ", final_decoder_out.shape)
        #----cnn decoder----

        decoder_out_rs  = final_decoder_out.view(-1,
                                                 self.hidden_dim,
                                                 self.hidden_spt,
                                                 self.hidden_spt)

        cnn_decoder_out_raw = torch.sigmoid(self.cnn_decoder(decoder_out_rs))
        cnn_decoder_out     = cnn_decoder_out_raw.view(self.batch_size,
                                                       y_seq_len,
                                                       self.out_channel,
                                                       self.img_size,
                                                       self.img_size)

        # print("cnn_decoder_out shape: ", cnn_decoder_out.shape)


        #----ouputs---------

        if step % self.print_step == 0 and is_train:
            output_str = " |     Iter: {0}    |\n".format(step)
            output_str += "-------------------------------------------\n"
            output_str += "   Sampling prob: {0:.3f}\n".format(sample_prob)
            output_str += "   Sampling:\n"
            output_str += "    " + str(r_hist) + "\n"
            output_str += "\n=================================================\n\n"
            print(output_str)

        return cnn_decoder_out
    
    def forward(self, x, y, step = 0, is_train = True):
        """
        Parameters
        ----------
        input_tensor:
            x: 5-D Tensor of shape (b, t_in, c_in, h, w)        #   batch, time, channel, height, width
            y: 5-D Tensor of shape (b, t_out, c_out, h, w)      #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, x_seq_len, _, _, _ = x.size()
        b, y_seq_len, _, _, _ = y.size()
        # print("x size: ", x.size())

        assert b == self.batch_size

        # autoencoder forward
        outputs = self.autoencoder(x, y, x_seq_len, y_seq_len, step = step, is_train = is_train)

        return outputs


def test():
    N, T, C, H, W = 32, 10, 5, 64, 64
    
    x = torch.randn((N, T, C, H, W ))
    y = torch.randn((N, T, 1, H, W ))

    model = EncoderDecoderConvLSTM(
                hidden_dim = 64,
                batch_size = 32,
                hidden_spt = 16,
                lstm_dims = [64, 64],
                img_size = 64,
                img_channel = 5,
                out_channel = 1,
                print_step = 500,
                alpha = 4000,
                beta = 16000,
                norm = nn.InstanceNorm2d,
                teacher_forcing = True,
    )

    output = model(x, y)

    assert output.shape == y.shape, "Test failed"
    # print("output shape: ",output.shape)

# test()
