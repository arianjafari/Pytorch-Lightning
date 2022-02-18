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
                img_channel = 1,
                out_channel = 1,
                print_step = 500,
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
        self.teacher_forcing = teacher_forcing

        
        self.cnn_encoder = ConvEncoder(in_channel = self.img_channel, out_dim = self.hidden_dim)
        self.cnn_decoder = ConvDecoder(b_size = self.batch_size, out_channel = self.out_channel, inp_dim = self.hidden_dim)
        
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
        alpha = 2450#1150
        beta  = 8000
        return alpha / (alpha + np.exp((step + beta) / alpha))

    def autoencoder(self, x, x_seq_len, future_step, step, is_train = True):
        
        nextf_raw = x[:, x_seq_len : , :, :, :] # the grand truth frames

        #----cnn encoder----
        # stack each sequence of B,T,C,H,W as BT,C,H,W to be fed in CNN model
        prevf_raw = x[:,:x_seq_len,:,:,:].contiguous().view(-1,self.img_channel,self.img_size,self.img_size)
        prevf_enc = self.cnn_encoder(prevf_raw).view(
                                                    self.batch_size,
                                                    x_seq_len,
                                                    self.hidden_dim,
                                                    self.hidden_spt,self.hidden_spt)

        if (is_train and self.teacher_forcing):
            cnn_encoder_out = self.cnn_encoder(nextf_raw.contiguous().view(-1,
                                                                           self.img_channel,
                                                                           self.img_size,
                                                                           self.img_size))
            nextf_enc       = cnn_encoder_out.view(
                                                   self.batch_size,
                                                   future_step,
                                                   self.hidden_dim,
                                                   self.hidden_spt,self.hidden_spt)

        #----lstm encoder---

        hidden           = self.lstm_encoder.get_init_states(self.batch_size)
        _, encoder_state = self.lstm_encoder(prevf_enc, hidden)

        #----lstm decoder---

        sample_prob =  self.get_sample_prob(step) if (is_train and self.teacher_forcing) else 0
        decoder_output_list = []
        r_hist = []

        for s in range(future_step):
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

        #----cnn decoder----

        decoder_out_rs  = final_decoder_out.view(-1,
                                                 self.hidden_dim,
                                                 self.hidden_spt,
                                                 self.hidden_spt)

        cnn_decoder_out_raw = torch.sigmoid(self.cnn_decoder(decoder_out_rs))
        cnn_decoder_out     = cnn_decoder_out_raw.view(self.batch_size,
                                                       future_step,
                                                       self.out_channel,
                                                       self.img_size,
                                                       self.img_size)


        #----ouputs---------

        if step % self.print_step == 0:
            output_str = " |     Iter: {0}    |\n".format(step)
            output_str += "-------------------------------------------\n"
            output_str += "   Sampling prob: {0:.3f}\n".format(sample_prob)
            output_str += "   Sampling:\n"
            output_str += "    " + str(r_hist) + "\n"
            output_str += "\n=================================================\n\n"
            print(output_str)

        return cnn_decoder_out
    
    def forward(self, x, future_seq=10, step = 0, is_train = True):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, xy_seq_len, _, h, w = x.size()
        x_seq_len = xy_seq_len - future_seq
        # print("x size: ", x.size())

        assert b == self.batch_size

        # autoencoder forward
        outputs = self.autoencoder(x, x_seq_len, future_seq, step = step, is_train = is_train)

        return outputs