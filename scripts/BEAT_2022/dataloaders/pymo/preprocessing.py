'''
Preprocessing Tranformers Based on sci-kit's API

By Omid Alemi
Created on June 12, 2017
'''
import copy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .Quaternions import Quaternions
from .rotation_tools import Rotation

class MocapParameterizer(BaseEstimator, TransformerMixin):
    def __init__(self, param_type = 'euler'):
        '''
        
        param_type = {'euler', 'quat', 'expmap', 'position'}
        '''
        self.param_type = param_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._to_expmap(X)
        elif self.param_type == 'quat':
            return X
        elif self.param_type == 'position':
            return self._to_pos(X)
        else:
            raise UnsupportedParamError('Unsupported param: %s. Valid param types are: euler, quat, expmap, position' % self.param_type)
#        return X
    
    def inverse_transform(self, X, copy=None): 
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._expmap_to_euler(X)
        elif self.param_type == 'quat':
            raise UnsupportedParamError('quat2euler is not supported')
        elif self.param_type == 'position':
            print('positions 2 eulers is not supported')
            return X
        else:
            raise UnsupportedParamError('Unsupported param: %s. Valid param types are: euler, quat, expmap, position' % self.param_type)

    def _to_pos(self, X):
        '''Converts joints rotations in Euler angles to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=euler_df.index)

            # Copy the root rotations into the new DataFrame
            # rxp = '%s_Xrotation'%track.root_name
            # ryp = '%s_Yrotation'%track.root_name
            # rzp = '%s_Zrotation'%track.root_name
            # pos_df[rxp] = pd.Series(data=euler_df[rxp], index=pos_df.index)
            # pos_df[ryp] = pd.Series(data=euler_df[ryp], index=pos_df.index)
            # pos_df[rzp] = pd.Series(data=euler_df[rzp], index=pos_df.index)

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)
            
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']
                #print("rot_order:" + joint + " :" + rot_order)

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values['%s_%srotation'%(joint, rot_order[0])], track.values['%s_%srotation'%(joint, rot_order[1])], track.values['%s_%srotation'%(joint, rot_order[2])]]))

                if pc.shape[1] < 3:
                    pos_values = np.asarray([[0,0,0] for f in pc.iterrows()])
                else:
                    pos_values =np.asarray([[f[1]['%s_Xposition'%joint], 
                                  f[1]['%s_Yposition'%joint], 
                                  f[1]['%s_Zposition'%joint]] for f in pc.iterrows()])
                
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)
                
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 
                if track.root_name == joint:
                    tree_data[joint][0] = quats#rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = tree_data[parent][0]*quats# np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])

                    # multiply k to the rotmat of the parent for every frame i
                    q = tree_data[parent][0]*k #np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = tree_data[parent][1] + q #q.reshape(k.shape[0],3) + tree_data[parent][1]

                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition'%joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Yposition'%joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Zposition'%joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)


            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q


    def _to_expmap(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            exp_df[rxp] = pd.Series(data=euler_df[rxp], index=exp_df.index)
            exp_df[ryp] = pd.Series(data=euler_df[ryp], index=exp_df.index)
            exp_df[rzp] = pd.Series(data=euler_df[rzp], index=exp_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                euler = [[f[1]['%s_Xrotation'%joint], f[1]['%s_Yrotation'%joint], f[1]['%s_Zrotation'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                exps = [Rotation(f, 'euler', from_deg=True).to_expmap() for f in euler] # Convert the eulers to exp maps
                
                # Create the corresponding columns in the new DataFrame
    
                exp_df['%s_alpha'%joint] = pd.Series(data=[e[0] for e in exps], index=exp_df.index)
                exp_df['%s_beta'%joint] = pd.Series(data=[e[1] for e in exps], index=exp_df.index)
                exp_df['%s_gamma'%joint] = pd.Series(data=[e[2] for e in exps], index=exp_df.index)

            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q

    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            euler_df = pd.DataFrame(index=exp_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            euler_df[rxp] = pd.Series(data=exp_df[rxp], index=euler_df.index)
            euler_df[ryp] = pd.Series(data=exp_df[ryp], index=euler_df.index)
            euler_df[rzp] = pd.Series(data=exp_df[rzp], index=euler_df.index)
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                euler_rots = [Rotation(f, 'expmap').to_euler(True)[0] for f in expmap] # Convert the eulers to exp maps
                
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_Xrotation'%joint] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_Yrotation'%joint] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_Zrotation'%joint] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q


class JointSelector(BaseEstimator, TransformerMixin):
    '''
    Allows for filtering the mocap data to include only the selected joints
    '''
    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            selected_joints.append(X[0].root_name)
        
        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            selected_channels.extend([o for o in X[0].values.columns if joint_name in o])
            
        Q = []


        for track in X:
            t2 = track.clone()
            
            for key in track.skeleton.keys():
                if key not in selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[selected_channels]

            Q.append(t2)
      

        return Q


class Numpyfier(BaseEstimator, TransformerMixin):
    '''
    Just converts the values in a MocapData object into a numpy array
    Useful for the final stage of a pipeline before training
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            Q.append(track.values.values)

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q

class RootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
        """
        self.method = method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[xpcol].diff()
                dz = track.values[zpcol].diff()    

                dx[0] = 0
                dz[0] = 0

                new_df.drop([xpcol, zpcol], axis=1, inplace=True)

                new_df[dxpcol] = dx
                new_df[dzpcol] = dz
                
                new_track = track.clone()
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name

                # Delta columns
                dxp_col = '%s_dXposition'%track.root_name
                dzp_col = '%s_dZposition'%track.root_name

                dxr_col = '%s_dXrotation'%track.root_name
                dyr_col = '%s_dYrotation'%track.root_name
                dzr_col = '%s_dZrotation'%track.root_name


                new_df = track.values.copy()

                root_pos_x_diff = pd.Series(data=track.values[xp_col].diff(), index=new_df.index)
                root_pos_z_diff = pd.Series(data=track.values[zp_col].diff(), index=new_df.index)

                root_rot_y_diff = pd.Series(data=track.values[yr_col].diff(), index=new_df.index)
                root_rot_x_diff = pd.Series(data=track.values[xr_col].diff(), index=new_df.index)
                root_rot_z_diff = pd.Series(data=track.values[zr_col].diff(), index=new_df.index)


                root_pos_x_diff[0] = 0
                root_pos_z_diff[0] = 0

                root_rot_y_diff[0] = 0
                root_rot_x_diff[0] = 0
                root_rot_z_diff[0] = 0

                new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[dxp_col] = root_pos_x_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df

            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        #TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0]-1):
                    recx.append(recx[i]+dx[i+1])
                    recz.append(recz[i]+dz[i+1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]

                new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)
                
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name

                # Delta columns
                dxp_col = '%s_dXposition'%track.root_name
                dzp_col = '%s_dZposition'%track.root_name

                dxr_col = '%s_dXrotation'%track.root_name
                dyr_col = '%s_dYrotation'%track.root_name
                dzr_col = '%s_dZrotation'%track.root_name


                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dz = track.values[dzp_col].values

                drx = track.values[dxr_col].values
                dry = track.values[dyr_col].values
                drz = track.values[dzr_col].values

                rec_xp = [startx]
                rec_zp = [startz]

                rec_xr = [0]
                rec_yr = [0]
                rec_zr = [0]


                for i in range(dx.shape[0]-1):
                    rec_xp.append(rec_xp[i]+dx[i+1])
                    rec_zp.append(rec_zp[i]+dz[i+1])
                    
                    rec_xr.append(rec_xr[i]+drx[i+1])
                    rec_yr.append(rec_yr[i]+dry[i+1])
                    rec_zr.append(rec_zr[i]+drz[i+1])


                new_df[xp_col] = pd.Series(data=rec_xp, index=new_df.index)
                new_df[zp_col] = pd.Series(data=rec_zp, index=new_df.index)

                new_df[xr_col] = pd.Series(data=rec_xr, index=new_df.index)
                new_df[yr_col] = pd.Series(data=rec_yr, index=new_df.index)
                new_df[zr_col] = pd.Series(data=rec_zr, index=new_df.index)

                new_df.drop([dxr_col, dyr_col, dzr_col, dxp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df

            Q.append(new_track)

        return Q


class RootCentricPositionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            for joint in all_but_root:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]-projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]-projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]-projected_root_pos[rzp], index=new_df.index)
            

            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]+projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]+projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]+projected_root_pos[rzp], index=new_df.index)
                

            new_track.values = new_df

            Q.append(new_track)
        
        return Q


class Flattener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)

class ConstantsRemover(BaseEstimator, TransformerMixin):
    '''
    For now it just looks at the first track
    '''

    def __init__(self, eps = 10e-10):
        self.eps = eps
        

    def fit(self, X, y=None):
        stds = X[0].values.std()
        cols = X[0].values.columns.values
        self.const_dims_ = [c for c in cols if (stds[c] < self.eps).any()]
        self.const_values_ = {c:X[0].values[c].values[0] for c in cols if (stds[c] < self.eps).any()}
        return self

    def transform(self, X, y=None):
        Q = []
        

        for track in X:
            t2 = track.clone()
            #for key in t2.skeleton.keys():
            #    if key in self.ConstDims_:
            #        t2.skeleton.pop(key)
            t2.values = track.values[track.values.columns.difference(self.const_dims_)]
            Q.append(t2)
        
        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            t2 = track.clone()
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]
            Q.append(t2)

        return Q

class ListStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, rate):
        self.rate = rate
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            #print(track.values.size)
            #new_track = track.clone()                            
            #new_track.values = track.values[0:-1:self.rate]            
            #print(new_track.values.size)
            new_track = track[0:-1:self.rate]
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X


#TODO: JointsSelector (x)
#TODO: SegmentMaker
#TODO: DynamicFeaturesAdder
#TODO: ShapeFeaturesAdder
#TODO: DataFrameNumpier (x)

class TemplateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

class UnsupportedParamError(Exception):
    def __init__(self, message):
        self.message = message
