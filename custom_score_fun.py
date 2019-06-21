from scipy.special import inv_boxcox
import numpy as np



def RMSLE(y_true, y_pred, transform, scale, mint):
    
    if transform == 'bc':
        #Inverse scaling
        y_pred = y_pred/scale['bc_trip_duration']+mint['bc_trip_duration']
        y_true = y_true/scale['bc_trip_duration']+mint['bc_trip_duration'] 

        #Inverse Box-Cox
        y_pred = inv_boxcox(y_pred,lambda_tripduration)
        y_true = inv_boxcox(y_true,lambda_tripduration)
    else:
        #Inverse scaling
        y_pred = y_pred/scale['log_trip_duration']+mint['log_trip_duration']
        y_true = y_true/scale['log_trip_duration']+mint['log_trip_duration'] 

        #Inverse Log
        y_pred = np.exp(y_pred)
        y_true = np.exp(y_true)
    
    e_i = np.square( np.log(y_pred + 1) - np.log( y_true + 1) )
    score = np.sqrt( (1/len(y_true)) * np.sum(e_i) )
    
    return score




