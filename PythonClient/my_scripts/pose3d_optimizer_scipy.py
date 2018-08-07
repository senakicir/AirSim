from helpers import * 
from project_bones import take_bone_projection
from torch.autograd import Variable

def mse_loss(input_1, input_2):
    return np.sum(np.square((input_1 - input_2))) / input_1.shape[0]

class pose3d_calibration_scipy():
    def __init__(self, model, data_list, weights, loss_dict, num_iterations):
        self.bone_connections, _, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        self.curr_iter = 0
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []


    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d, R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)

        left_bone_connections, right_bone_connections, _ = split_bone_connections(self.bone_connections)

        bonelosses = np.zeros([len(left_bone_connections),1])
        for i, l_bone in enumerate(left_bone_connections):
            r_bone = right_bone_connections[i]
            left_length_of_bone = (np.sum(np.square(pose_3d[:, l_bone[0]] - pose_3d[:, l_bone[1]])))
            right_length_of_bone = (np.sum(np.square(pose_3d[:, r_bone[0]] - pose_3d[:, r_bone[1]])))
            bonelosses[i] = np.square((left_length_of_bone - right_length_of_bone))
        output["sym"] = np.sum(bonelosses)/bonelosses.shape[0]

        #if (self.curr_iter % 5000 == 0):
            #print("output", output)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        self.curr_iter += 1
        return overall_output
    

class pose3d_flight_scipy():
    def __init__(self, model, data_list, lift_list, weights, loss_dict, num_iterations, window_size, bone_lengths_):
        self.bone_connections, self.joint_names, self.NUM_OF_JOINTS, _ = model_settings(model)
        self.data_list = data_list
        self.lift_list = lift_list
        self.energy_weights = weights
        self.pltpts = {}
        self.loss_dict = loss_dict
        self.window_size = window_size
        self.bone_lengths = bone_lengths_
        self.curr_iter = 0
        for loss_key in self.loss_dict:
            self.pltpts[loss_key] = []


    def forward(self, pose_3d):
        pose_3d = np.reshape(a = pose_3d, newshape = [self.window_size, 3, self.NUM_OF_JOINTS], order = "C")

        output = {}
        for loss_key in self.loss_dict:
            output[loss_key] = 0

        #projection
        queue_index = 0
        for bone_2d_, R_drone_, C_drone_ in self.data_list:
            projected_2d, _ = take_bone_projection(pose_3d[queue_index, :, :], R_drone_, C_drone_)
            output["proj"] += mse_loss(projected_2d, bone_2d_)

            #smoothness
            if (queue_index != self.window_size-1 and queue_index != 0):
                output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :]) +  mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])
            elif (queue_index != self.window_size-1 ):
                output["smooth"] += mse_loss(pose_3d[queue_index, :, :], pose_3d[queue_index+1, :, :])
            elif (queue_index != 0):
                output["smooth"] += mse_loss(pose_3d[queue_index-1, :, :], pose_3d[queue_index, :, :])

            #smooth pose
            hip_index = self.joint_names.index('spine1')
            hip = pose_3d[queue_index, :, hip_index]
            temp_pose3d_t = pose_3d[queue_index, :, :] - hip[:, np.newaxis]
            if (queue_index != self.window_size-1 and queue_index != 0):
                p_hip = pose_3d[queue_index+1, :, hip_index]
                temp_pose3d_t_p_1 = pose_3d[queue_index+1, :, :]- p_hip[:, np.newaxis]
                m_hip = pose_3d[queue_index-1, :, hip_index]
                temp_pose3d_t_m_1 = pose_3d[queue_index-1, :, :]- m_hip[:, np.newaxis]
                output["smoothpose"] += mse_loss(temp_pose3d_t, temp_pose3d_t_p_1) +  mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)
            elif (queue_index != self.window_size-1 ):
                p_hip = pose_3d[queue_index+1, :, hip_index]
                temp_pose3d_t_p_1 = pose_3d[queue_index+1, :, :]- p_hip[:, np.newaxis]
                output["smoothpose"] += mse_loss(temp_pose3d_t, temp_pose3d_t_p_1)
            elif (queue_index != 0):
                m_hip = pose_3d[queue_index-1, :, hip_index]
                temp_pose3d_t_m_1 = pose_3d[queue_index-1, :, :]- m_hip[:, np.newaxis]
                output["smoothpose"] += mse_loss(temp_pose3d_t_m_1, temp_pose3d_t)

            #lift
            pose3d_lift = self.lift_list[queue_index]
            max_z = np.max(temp_pose3d_t[2,:])
            min_z = np.min(temp_pose3d_t[2,:])
            normalized_pose_3d = (pose3d_lift-min_z)/(max_z - min_z)
            output["lift"]= mse_loss(pose3d_lift, normalized_pose_3d)

            #bone length consistency 
            bonelosses = np.zeros([self.NUM_OF_JOINTS-1,1])
            for i, bone in enumerate(self.bone_connections):
                length_of_bone = (np.sum(np.square(pose_3d[queue_index, :, bone[0]] - pose_3d[queue_index, :, bone[1]])))
                bonelosses[i] = np.square((self.bone_lengths[i] - length_of_bone))
            output["bone"] = np.sum(bonelosses)/(self.NUM_OF_JOINTS-1)

            queue_index += 1


        if (self.curr_iter % 5000 == 0):
            print("output", output)

        overall_output = 0
        for loss_key in self.loss_dict:
            overall_output += self.energy_weights[loss_key]*output[loss_key]/len(self.loss_dict)
            self.pltpts[loss_key].append(output[loss_key])
        
        self.curr_iter += 1
        return overall_output
    
def minimize_levenberg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=1e-6, maxiter=None,
                       disp=False, return_all=False,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.
    Note that the `jac` parameter (Jacobian) is required.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    f = fun
    fprime = jac
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0_flat = asarray(x0).flatten()
    fcalls, f = wrap_function(f, args)
    gcalls, fprime = wrap_function(fprime, args)
    hcalls = 0
    if maxiter is None:
        maxiter = len(x0_flat)*200

    xtol = len(x0_flat) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
    warnflag = 0
    while (np.add.reduce(np.abs(update)) > xtol) and (k < maxiter):
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = fhess(*(xk,) + args)
            hcalls = hcalls + 1

        while np.add.reduce(np.abs(ri)) > termcond:
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                Ap = np.dot(A, psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1          # update numpy.dot(ri,ri) for next time.

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % fcalls[0])
            print("         Gradient evaluations: %d" % gcalls[0])
            print("         Hessian evaluations: %d" % hcalls)
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % fcalls[0])
            print("         Gradient evaluations: %d" % gcalls[0])
            print("         Hessian evaluations: %d" % hcalls)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % fcalls[0])
            print("         Gradient evaluations: %d" % gcalls[0])
            print("         Hessian evaluations: %d" % hcalls)

    result = OptimizeResult(fun=fval, jac=gfk, nfev=fcalls[0], njev=gcalls[0],
                            nhev=hcalls, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
