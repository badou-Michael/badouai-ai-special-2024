import numpy as np
import scipy as sp
import scipy.linalg as sl

def random_index(n,n_data):
    index=np.arange(n_data)
    np.random.shuffle(index)
    index1=index[:n]
    index2=index[n:]
    return index1, index2

class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    
    def fit(self, data):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T 
        x, resids, rank, s = sl.lstsq(A, B) 
        return x  
 
    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T 
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T 
        B_fit = np.dot(A, model) 
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) 
        return err_per_point

def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    iterations = 0
    bestfit = None
    besterr = np.inf 
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_index(n, data.shape[0])
        #print ('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs] 
        maybemodel = model.fit(maybe_inliers) 
        test_err = model.get_error(test_points, maybemodel)
        #print('test_err = ', test_err <t)
        also_idxs = test_idxs[test_err < t]
        #print ('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs,:]
        #if debug:
            #print ('test_err.min()',test_err.min())
            #print ('test_err.max()',test_err.max())
            #print ('numpy.mean(test_err)',numpy.mean(test_err))
            #print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        # if len(also_inliers > d):
        #print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) 
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) 
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) 
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit


   
def test():
    n=500
    x=20*np.random.random((n,1)) #matrix 500x1
    k=60*np.random.normal(size=(1,1))
    B_data=np.dot(x,k)
    x_noise=x+2*np.random.normal( size = x.shape )#larger noise
    B_noise=B_data+2*np.random.normal( size = B_data.shape )

    if 1:
        n_other=100
        index=np.arange(x_noise.shape[0])
        np.random.shuffle(index)
        index=index[:n_other]
        x_noise[index]=20*np.random.random((n_other,1))
        B_noise[index]=50*np.random.normal(size=(n_other,1))
        print (B_noise[index])

    all_data=np.hstack((x_noise,B_noise))
    input_columns=range(1)
    output_columns=[1]
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug)
    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 2000, 7e3, 300, debug = debug, return_all = True)
 
    if 1:
        import pylab
 
        sort_idxs = np.argsort(x[:,0])
        A_col0_sorted = x[sort_idxs] 
 
        if 1:
            pylab.plot( x_noise[:,0], B_noise[:,0], 'k.', label = 'data' ) 
            pylab.plot( x_noise[ransac_data['inliers'], 0], B_noise[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" )
        else:
            pylab.plot( x_noise[non_outlier_idxs,0], B_noise[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( x_noise[outlier_idxs,0], B_noise[outlier_idxs,0], 'r.', label='outlier data' )
 
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,k)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()
        
 
if __name__ == "__main__":
    test()


   
 
    
