# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:06:24 2022

@author: josh
"""

import numpy as np
import os
#import matplotlib.pyplot as plt

#os.chdir('C:\\Users\\josh\\Documents\\CHON_EOS_table')

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#%%          Input parameters for the training process

rho0 = 2.1                           # in g/cm^3
e0 = 100                             # in eV
M = (1.67356 + 19.94473)*10.0**(-27) # in kg, this is mass of CH pair

prob = 0.9                           # probability each sample is used for on a given epoch

lam1 = 0.3                           # hyperparameters in the cost function associated with fractional energy
lam2 = 0.3                           # hyperparameters in the cost function associated with fractional pressure

x1 = 0.0                             # next 4 quantities are used to turn on penalty for negative energy
x2 = 0.0                             # this penalty was not used in prior version of the model
y1 = 2.0
y2 = 2.0


Nh = 80                              # number of of nodes in the hidden layer
D_in = 2                             # dimension of the input layer
D_out = 1                            # dimension of the output layer
                                     # Note: currently code does not support the use of a deep NN.
p_start = 0.0                        # minimum value in the initialization of the weights and bias
p_end = 0.05                         # maximum value in the initialization of the weights and bias

lr = 0.003                           # learning rate
epochs = 400000                      # number of training cycles 


#%%
def act(x_in):                                                            # activation function
    return np.tanh( x_in )

def d_act(y_in):                                                          # derivative of the activation function
    return np.subtract(1.0, np.square( np.tanh(y_in) ) )

def d2_act(z_in):                                                         # second derivative of the activation function
    hold1 = np.multiply( -2.0, np.tanh(z_in))
    hold2 = d_act(z_in)
    return np.multiply(hold1, hold2)


def initial(row, column, fp_min, fp_max):                                # sets up matricies of random number, used to initialize weights and bias
    new_matrix = []  
    for ii in range(0,row):
        w_i=[np.random.uniform(fp_min, fp_max) for jj in range(0,column)]
        new_matrix.append(w_i)
    return np.matrix(new_matrix)


def data_load(file, scale_parms):                                        # Loads in the reference data
                                                                         # Note: all data scaling happens here
    E0 = scale_parms[0]
    R0 = scale_parms[1]
    m = scale_parms[2]
    
    P0 = R0*E0/m*1.602*10**(-25)         # if R0 g/cm^3, m kg and E0 eV then P0 in GPa
    
    
    Ha2eV = 27.2114                      # 1 Ha = 27.2114 eV
    kb = 0.00008617333                   # eV/K
    
    In = []
    Weight = []
    Pressure = []
    Energy = []
    
    with open(file, 'r') as infile:
        for row in infile:
            
            hold = [xx for xx in row.split(' ') if xx != '\n' and xx != '']
            
            dens = float( hold[5] )/R0                                      # density in g/cm^3
            press = float( hold[11] )/P0                                    # pressure is now in eV/A^3
            temp = kb*float( hold[9] )/E0                                   # temp is in kelven converted to unitless variable
            energy = float( hold[-2] )*Ha2eV/E0                             # energy is now eV then shifted and made dimensionless
  

            In.append( [ np.log(temp), np.log(dens) ] )
            Weight.append(  [energy*dens/press, dens/press] )
            Pressure.append( press )
            Energy.append( energy )
          
    return np.transpose( np.matrix(In) ), np.transpose( np.matrix( Weight ) ), Pressure, Energy
    

def weight_bias_out(weight_list,bias_list, weight_filename, bias_filename):
    
    
    '''
    Writes out the weights and biases of the NN.
    weight_list: list of weight matricies 
    bias_list: list of bias matrices (note: currently all columns are the same)
    NOTE: each row in the output file is the ROW of a weight matrix
    NOTE: each row in the output file is the bias COLUMN vector for the layer 
    '''
    with open(weight_filename,'w') as weight_out:
        
        weight_out.write('number of weights = ' + str(len(weight_list)) + '\n')
        
        for ii in range(0,len(weight_list)):
            Wgt_out = weight_list[ii]
            n_rows, n_col = np.shape(Wgt_out)
            weight_out.write('layer ' + str(ii) + ' n_rows = ' + str(n_rows) + ' n_col = ' + str(n_col) + '\n')
            
            for jj in range( n_rows ):
                for kk in range( n_col ):
                    weight_out.write(str(Wgt_out[jj,kk]))
                    if kk != n_col-1:
                        weight_out.write(' , ')
                weight_out.write('\n')
                
    with open(bias_filename,'w') as bias_out:
        
        bias_out.write('number of bias vectors = ' + str(len( bias_list )) + '\n')
        
        for ii in range(len( bias_list )):
            Bs_out = bias_list[ii]
            n_rows, n_col = np.shape( Bs_out ) 
            bias_out.write('layer ' + str(ii) + ' nrows = ' + str(n_rows) + '\n')
            
            for jj in range( n_rows ):
                bias_out.write( str(Bs_out[jj,0]) )
                if jj != n_rows-1:
                    bias_out.write(' , ')
            bias_out.write('\n')

def WEIGHT_in(filename):
    wgt_list = []
    
    with open(filename,'r') as weight_in:
        i_index = 0
        Wgh_mat_hold = []
        Wgh_row_hold = []
        
        for row in weight_in:
            i_index = i_index + 1
            
            if row.find('number') < 0 and row.find('layer') < 0:
                Wgh_row_hold=[float( elem ) for elem in row.split(',')]
                Wgh_mat_hold.append( Wgh_row_hold )
            
            if row.find('layer') >= 0 and i_index > 2:
                wgt_list.append( np.matrix( Wgh_mat_hold ) )
                Wgh_mat_hold = []
                
        wgt_list.append( np.matrix(Wgh_mat_hold) )
        
    return wgt_list


def BIAS_in(filename):
    bs_list = []
    
    with open(filename,'r') as bias_in:
        Bs_vec = []
        i_index = 0
        
        for row in bias_in:
            i_index = i_index + 1
            
            if row.find('number') < 0 and row.find('layer') < 0:
                Bs_vec = [float( elem ) for elem in row.split(',')]
            
            if row.find('layer') >= 0 and i_index > 2:
                bs_list.append( np.transpose( np.matrix( Bs_vec ) ) )
                Bs_vec = []
                
        bs_list.append( np.transpose( np.matrix( Bs_vec ) ) )
        
    return bs_list


def gradient_check(current_epoch, grad_W1, grad_W2, grad_B, grad_W1_con, grad_W2_con, grad_B_con):


    with open('grad_check.txt', 'a') as GC_out:
        
        GC_out.write('epoch ' + str(current_epoch))
        GC_out.write(' dC_dW1 ' + str(np.max( np.fabs(grad_W1) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_W1) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_W1) )) )
        
        GC_out.write(' dC_dW2 ' + str(np.max( np.fabs(grad_W2) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_W2) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_W2) )) )   
    
        GC_out.write(' dC_dB ' + str(np.max( np.fabs(grad_B) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_B) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_B) )) )
    
        GC_out.write(' dC_dW1_con ' + str(np.max( np.fabs(grad_W1_con) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_W1_con) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_W1_con) )) )
        
        GC_out.write(' dC_dW2_con ' + str(np.max( np.fabs(grad_W2_con) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_W2_con) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_W2_con) )) )   
    
        GC_out.write(' dC_dB_con ' + str(np.max( np.fabs(grad_B_con) )) )
        GC_out.write(' ' + str(np.min( np.fabs(grad_B_con) )) )
        GC_out.write(' ' + str(np.average( np.fabs(grad_B_con) )) )
    
        GC_out.write('\n')
    

#%%        Data is read in and transformed here




if rank == 0:

    Inp_tr, Wvec_tr, P_tr, E_tr = data_load( 'train_data.txt', [e0, rho0, M])
    Inp_val, Wvec_val, P_val, E_val = data_load( 'val_data.txt', [e0, rho0, M])
        
    ic = np.random.randint( 0, np.shape(Inp_tr)[1])


else:
    Inp_tr, Wvec_tr, P_tr, E_tr = [], [], [], []
    Inp_val, Wvec_val, P_val, E_val = [], [], [], []
    ic = []
    
Inp_tr = comm.bcast( Inp_tr, root = 0)
Wvec_tr = comm.bcast( Wvec_tr, root = 0)
P_tr = comm.bcast(P_tr, root = 0)
E_tr = comm.bcast(E_tr, root = 0)

Inp_val = comm.bcast( Inp_val, root = 0)
Wvec_val = comm.bcast( Wvec_val, root = 0)
P_val = comm.bcast(P_val, root = 0)
E_val = comm.bcast(E_val, root = 0)

init_cond_index = comm.bcast(ic, root = 0)


Ntot_tr = np.shape(Inp_tr)[1]                      # CH data has 198 total data points
sub_index_tr = [ii*size + rank for ii in range(Ntot_tr) if (ii*size + rank) < Ntot_tr ]


Ntot_val = np.shape(Inp_val)[1]                      # CH data has 198 total data points
sub_index_val = [ii*size + rank for ii in range(Ntot_val) if (ii*size + rank) < Ntot_val ]


#%%          Setting up the initial parameters of the model    


# if an initial file for the weights and bias are not found then a random initialization will be done

if rank == 0: 
    
    if os.path.isfile('weights_0.txt'):
        W_hold = WEIGHT_in('weights_0.txt')
        W1_init = W_hold[0]
        W2_init = W_hold[1]
    else:
        W1_init = initial(Nh, D_in, p_start, p_end)
        W2_init = initial(D_out, Nh, p_start, p_end)
            
    if os.path.isfile('bias_0.txt'):
        Bi_init = BIAS_in('bias_0.txt')[0]
    else:
        Bi_init = initial(Nh, 1, p_start, p_end)
        
        
        
    if os.path.isfile('cost_log.txt'):
        
        line_count = 0
        with open('cost_log.txt', 'r') as cin:
            for row in cin:
                line_count = line_count + 1
        
        epochs = epochs - line_count
        
    else:
        pass
    
else:
    W1_init = []
    W2_init = []
    Bi_init = []
    
    
W1_init = comm.bcast( W1_init, root = 0)
W2_init = comm.bcast( W2_init, root = 0)
Bi_init = comm.bcast( Bi_init, root = 0) 
  
epochs = comm.bcast( epochs, root = 0)

Cost = []

#%%              Training the model

W1 = W1_init
W2 = W2_init
Bi = Bi_init


avec = np.transpose( np.matrix([1.0, 0.0]))
avec_t = np.transpose( avec )

bvec = np.transpose( np.matrix([0.0, 1.0]))
bvec_t = np.transpose( bvec )

for step in range( epochs ):
    
    c_step = 0.0
    c_step_val = 0.0
    
    W2t = np.transpose( W2 )
    W1t = np.transpose( W1 )
    
    
    dC_dW2 = 0.0
    dC_dW1 = 0.0
    dC_dB = 0.0

    C0 = 0.0
    C0_val = 0.0
    dW2_0 = 0.0
    dW1_0 = 0.0
    dB_0 = 0.0    
 
    ns = 0
    
    for samp_num in sub_index_tr:
        
        if np.random.uniform(0, 1.0) < prob:
            ns = ns + 1
            
            I = Inp_tr[ :, samp_num ]
            It = np.transpose( I )
            
            wvec = Wvec_tr[:, samp_num]
            ERP = wvec[0,0]
            RP = wvec[1,0]
            
            P_0 = P_tr[ samp_num ]
            E_0 = E_tr[ samp_num ]            
            
            H1 = W1*I + Bi
            G = act( H1 )
            Gt = np.transpose( G )
            Gp = d_act( H1 )
            Gpt = np.transpose( Gp )
            Gpp = d2_act( H1 )

            
            df_dI = W1t*np.multiply(W2t, Gp)
            f = (W2*G)[0,0]
            F = np.sinh(f)

            
            nu = ERP - F*RP
            v_vec = avec + nu*bvec
            v_vect = np.transpose(v_vec)
            
            
            pde = (v_vect*df_dI)[0,0]
            
            # backprop contribution from the pde cost
            dC_dW2 = dC_dW2 + pde*(-1.0*RP*np.cosh(f)*bvec_t*df_dI*Gt + np.multiply(v_vect*W1t, Gpt) )
            
            dC_dW1 = dC_dW1 + pde*(-1.0*RP*np.cosh(f)*(bvec_t*df_dI)[0,0]*np.multiply(W2t, Gp)*It + np.multiply(W2t, Gp)*v_vect + np.multiply(Gpp, np.multiply(W2t, W1*v_vec))*It ) 
            
            dC_dB = dC_dB + pde*(-1.0*RP*np.cosh(f)*(bvec_t*df_dI)[0,0]*np.multiply(W2t, Gp) + np.multiply(Gpp, np.multiply(W2t, W1*v_vec)) ) 
            

            E_ml = F - np.sqrt(F**2 + 1)*df_dI[0,0]
            P_ml = np.exp(I[1,0])*np.sqrt(F**2 + 1)*df_dI[1,0]
            
            # coefficents for the pentalty based on 1-E/E0 and 1-P/P0
            Ce1 = 2.0*lam1*(1.0 -E_ml/E_0)*(-1.0/E_0)*(1.0 - F*df_dI[0,0]/np.sqrt(F**2+1.0) )*np.cosh(f)
            Ce2 = 2.0*lam1*(1.0 -E_ml/E_0)*(-1.0/E_0)*np.sqrt(F**2+1.0)
            
            Cp1 = 2.0*lam2*(1.0 -P_ml/P_0)*(-1.0/P_0)*F*np.cosh(f)*df_dI[1,0]*np.exp(I[1,0])/np.sqrt(F**2+1.0)
            Cp2 = 2.0*lam2*(1.0 -P_ml/P_0)*(-1.0/P_0)*np.exp(I[1,0])*np.sqrt(F**2+1.0)
            
            # penalties for having the wrong sign on gradients
            S1 = np.exp( x1*( (avec_t*df_dI)[0,0] + y1) )
            S2 = np.exp( -1.0*x2*( (bvec_t*df_dI)[0,0] -y2) )
            
            rvec = (-Ce2 + x1*S1)*avec + (Cp2 - x2*S2)*bvec
            rvec_t = np.transpose( rvec )
            
            #backprop contribution from penalty terms
            dW2_0 = dW2_0 + (Ce1 + Cp1)*Gt + np.multiply(rvec_t*W1t, Gpt)
            dW1_0 = dW1_0 + (Ce1 + Cp1)*np.multiply(W2t, Gp)*It + np.multiply(W2t, Gp)*rvec_t + np.multiply(W1*rvec, np.multiply(W2t, Gpp))*It
            dB_0 = dB_0 + (Ce1 + Cp1)*np.multiply(W2t, Gp) + np.multiply(W1*rvec, np.multiply(W2t, Gpp))
    
            # total cost associated with the current training point   
            c_step = c_step + pde**2 
            C0 = C0 + lam1*(1.0 - E_ml/E_0)**2 + lam2*(1.0 - P_ml/P_0)**2
        
            # contribution to the backpropogation from the current training point
            #dW2 = dW2 + dC_dW2 + dW2_0
            #dW1 = dW1 + dC_dW1 + dW1_0
            #dB = dB + dC_dB + dB_0
            
        else:
            pass

        
    for samp_num in sub_index_val:
             
        I = Inp_val[ :, samp_num ]
        It = np.transpose( I )
        
        wvec = Wvec_val[:, samp_num]
        ERP = wvec[0,0]
        RP = wvec[1,0]
        
        P_0 = P_val[ samp_num ]
        E_0 = E_val[ samp_num ]            
        
        H1 = W1*I + Bi
        G = act( H1 )
        Gt = np.transpose( G )
        Gp = d_act( H1 )
        Gpt = np.transpose( Gp )
        Gpp = d2_act( H1 )

        
        df_dI = W1t*np.multiply(W2t, Gp)
        f = (W2*G)[0,0]
        F = np.sinh(f)

        E_ml = F - np.sqrt(F**2 + 1)*df_dI[0,0]
        P_ml = np.exp(I[1,0])*np.sqrt(F**2 + 1)*df_dI[1,0]

        
        nu = ERP - F*RP
        v_vec = avec + nu*bvec
        v_vect = np.transpose(v_vec)
        
        
        pde = (v_vect*df_dI)[0,0]
        
        c_step_val = c_step_val + pde**2 
        
        C0_val = C0_val + lam1*(1.0 - E_ml/E_0)**2 + lam2*(1.0 - P_ml/P_0)**2


    c_hold_pde = comm.gather( c_step, root = 0)
    c_hold_0 = comm.gather( C0, root = 0)
    c_holdval_0 = comm.gather( C0_val, root = 0)
    
    c_hold_pde_val = comm.gather( c_step_val, root = 0)
    
    dC_dW1_hold = comm.gather( dC_dW1, root = 0)
    dC_dW2_hold = comm.gather( dC_dW2, root = 0)
    dC_dB_hold = comm.gather( dC_dB, root = 0)
    
    dC_dW1_0hold = comm.gather( dW1_0, root = 0)
    dC_dW2_0hold = comm.gather( dW2_0, root = 0)
    dC_dB_0hold = comm.gather( dB_0, root = 0)
    
    ns = comm.gather(ns, root = 0)

    if rank == 0:
        #if step%100 == 0:
        #    ic = np.random.randint( 0, np.shape(Inp_tr)[1])
        #else:
        #    pass
        
        Ns = np.sum( ns )
        C_pde = np.sum( c_hold_pde )/Ns
        #C0 = np.sum( c_hold_0 )
        C0 = np.sum( c_hold_0 )/Ns
        
        C_pde_val = np.sum( c_hold_pde_val )/Ntot_val
        C_0_val = np.sum( c_holdval_0 )/Ntot_val
        
        dW1 = 0.0
        dW2 = 0.0
        dB = 0.0
        
        dW1_0 = 0.0
        dW2_0 = 0.0
        dB_0 = 0.0
        
        
        for ii in range(size):
            dW1 = dW1 + dC_dW1_hold[ii]
            dW2 = dW2 + dC_dW2_hold[ii]
            dB = dB + dC_dB_hold[ii] 
            
            dW1_0 = dW1_0 + dC_dW1_0hold[ii]
            dW2_0 = dW2_0 + dC_dW2_0hold[ii]
            dB_0 = dB_0 + dC_dB_0hold[ii] 
            
        
        W2 = W2 - lr/Ns*(dW2 + dW2_0)
        W1 = W1 - lr/Ns*(dW1 + dW1_0)
        Bi = Bi - lr/Ns*(dB + dB_0)


        with open('cost_log.txt','a') as ofil:
            ofil.write('epoch ' + str(step) + ' cost_pde ' + str(C_pde) + ' cost_0 '+ str(C0) + ' cost_val '+ str(C_pde_val) + ' cost_0_val '+ str(C_0_val) +'\n' )
        if step % 1000 == 0:    
            weight_bias_out([W1, W2], [Bi], 'weights.txt', 'bias.txt')
            gradient_check(step, dW1, dW2, dB, dW1_0, dW2_0, dB_0) 
    else:
        pass
    
    W1 = comm.bcast(W1, root = 0)
    W2 = comm.bcast(W2, root = 0)
    Bi = comm.bcast(Bi, root = 0)
        
    #init_cond_index = comm.bcast(ic, root = 0)



#%%                 test set

if rank == 0:
    Inp_te, Wvec_te, P_te, E_te = data_load( 'test_data.txt', [e0, rho0, M])
    Ntest = np.shape(Inp_te)[1]
    
    Press_err = []
    Energy_err = []
    dlnR = []
    dlnT = []
    F_list = []
    P_ref_list = []
    E_ref_list = []
    P_ml_list = []
    E_ml_list = []
    ERP = []
    
    for samp_num in range(Ntest):
        
        I = Inp_te[ :, samp_num ]
        wvec = Wvec_te[ :, samp_num]
        
        H = W1*I + Bi
        G = act( H )
        Gp = d_act( H )  
        
        f = (W2*G)[0,0]
        F = np.sinh(f)
        df_dI = np.transpose(W1)*np.multiply( np.transpose(W2), Gp)  
        
        E_ml = F - np.sqrt(F**2 + 1)*df_dI[0,0]
        P_ml = np.exp(I[1,0])*np.sqrt(F**2 + 1)*df_dI[1,0]      
        
        P_ref = P_te[ samp_num ]
        E_ref = E_te[ samp_num ]
                
        Press_err.append( np.fabs( (P_ref - P_ml)/P_ref )*100.0 )
        Energy_err.append( np.fabs( (E_ref - E_ml)/E_ref )*100.0 )
        
        F_list.append(F)
        dlnR.append( df_dI[1,0] )
        dlnT.append( df_dI[0,0] )
        P_ref_list.append( P_ref )
        P_ml_list.append( P_ml )
        E_ref_list.append( E_ref )
        E_ml_list.append( E_ml )
        ERP.append( wvec[1,0] )
        
    with open('val_log.txt','a') as ofil:
        ofil.write('relErr_pressure ' + str( np.percentile(Press_err, [10, 25, 50, 75, 90]) ) + '\n' )
        ofil.write('relErr_energy ' + str( np.percentile(Energy_err, [10, 25, 50, 75, 90]) ) + '\n' )
        ofil.write('Ref_press ' + str( P_ref_list ) +'\n')
        ofil.write('ML_press ' + str( P_ml_list ) +'\n')
        ofil.write('Ref_energy ' + str( E_ref_list ) +'\n')
        ofil.write('ML_energy ' + str( E_ml_list ) +'\n')
        ofil.write('dlnR ' + str( dlnR ) +'\n')
        ofil.write('dlnT ' + str( dlnT ) +'\n')
        ofil.write('ERP ' + str( ERP ) +'\n')
        ofil.write('F ' + str( F_list ) +'\n')
    
    










