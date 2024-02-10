'''
Tools for Manipulating and Converting 3D Rotations

By Omid Alemi
Created: June 12, 2017

Adapted from that matlab file...
'''

import math
import numpy as np

def deg2rad(x):
    return x/180*math.pi


def rad2deg(x):
    return x/math.pi*180

class Rotation():
    def __init__(self,rot, param_type, rotation_order, **params):
        self.rotmat = []
        self.rotation_order = rotation_order
        if param_type == 'euler':
            self._from_euler(rot[0],rot[1],rot[2], params)
        elif param_type == 'expmap':
            self._from_expmap(rot[0], rot[1], rot[2], params)

    def _from_euler(self, alpha, beta, gamma, params):
        '''Expecting degress'''

        if params['from_deg']==True:
            alpha = deg2rad(alpha)
            beta = deg2rad(beta)
            gamma = deg2rad(gamma)
        
        ca = math.cos(alpha)
        cb = math.cos(beta)
        cg = math.cos(gamma)
        sa = math.sin(alpha)
        sb = math.sin(beta)
        sg = math.sin(gamma)        

        Rx = np.asarray([[1, 0, 0], 
              [0, ca, sa], 
              [0, -sa, ca]
              ])

        Ry = np.asarray([[cb, 0, -sb], 
              [0, 1, 0],
              [sb, 0, cb]])

        Rz = np.asarray([[cg, sg, 0],
              [-sg, cg, 0],
              [0, 0, 1]])

        self.rotmat = np.eye(3)

        ############################ inner product rotation matrix in order defined at BVH file #########################
        for axis in self.rotation_order :
            if axis == 'X' :
                self.rotmat = np.matmul(Rx, self.rotmat)
            elif axis == 'Y':
                self.rotmat = np.matmul(Ry, self.rotmat)
            else :
                self.rotmat = np.matmul(Rz, self.rotmat)
        ################################################################################################################
   
    def _from_expmap(self, alpha, beta, gamma, params):
        if (alpha == 0 and beta == 0 and gamma == 0):
            self.rotmat = np.eye(3)
            return

        #TODO: Check exp map params

        theta = np.linalg.norm([alpha, beta, gamma])

        expmap = [alpha, beta, gamma] / theta

        x = expmap[0]
        y = expmap[1]
        z = expmap[2]

        s = math.sin(theta/2)
        c = math.cos(theta/2)

        self.rotmat = np.asarray([
            [2*(x**2-1)*s**2+1,  2*x*y*s**2-2*z*c*s,  2*x*z*s**2+2*y*c*s],
            [2*x*y*s**2+2*z*c*s,  2*(y**2-1)*s**2+1,  2*y*z*s**2-2*x*c*s],
            [2*x*z*s**2-2*y*c*s, 2*y*z*s**2+2*x*c*s , 2*(z**2-1)*s**2+1]
        ])
        


    def get_euler_axis(self):
        R = self.rotmat
        theta = math.acos((self.rotmat.trace() - 1) / 2)
        axis = np.asarray([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        axis = axis/(2*math.sin(theta))
        return theta, axis

    def to_expmap(self):
        theta, axis = self.get_euler_axis()
        rot_arr = theta * axis
        if np.isnan(rot_arr).any():
            rot_arr = [0, 0, 0]
        return rot_arr
    
    def to_euler(self, use_deg=False):
        eulers = np.zeros((2, 3))

        if np.absolute(np.absolute(self.rotmat[2, 0]) - 1) < 1e-12:
            #GIMBAL LOCK!
            print('Gimbal')
            if np.absolute(self.rotmat[2, 0]) - 1 < 1e-12:
                eulers[:,0] = math.atan2(-self.rotmat[0,1], -self.rotmat[0,2])
                eulers[:,1] = -math.pi/2
            else:
                eulers[:,0] = math.atan2(self.rotmat[0,1], -elf.rotmat[0,2])
                eulers[:,1] = math.pi/2
            
            return eulers

        theta = - math.asin(self.rotmat[2,0])
        theta2 = math.pi - theta

        # psi1, psi2
        eulers[0,0] = math.atan2(self.rotmat[2,1]/math.cos(theta), self.rotmat[2,2]/math.cos(theta))
        eulers[1,0] = math.atan2(self.rotmat[2,1]/math.cos(theta2), self.rotmat[2,2]/math.cos(theta2))

        # theta1, theta2
        eulers[0,1] = theta
        eulers[1,1] = theta2

        # phi1, phi2
        eulers[0,2] = math.atan2(self.rotmat[1,0]/math.cos(theta), self.rotmat[0,0]/math.cos(theta))
        eulers[1,2] = math.atan2(self.rotmat[1,0]/math.cos(theta2), self.rotmat[0,0]/math.cos(theta2))

        if use_deg:
            eulers = rad2deg(eulers)

        return eulers
    
    def to_quat(self):
        #TODO
        pass
    
    def __str__(self):
        return "Rotation Matrix: \n " + self.rotmat.__str__()
    



