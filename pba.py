

def cBox(k, n, nsteps = 1000):
    
    tol = 1e-06
    
    import numpy as np
    
    from scipy.stats import beta as beta
    
    if k == 0:
        k = tol
    
    else:
        k = k
        
    if k == [0, 0]:
        
        k = [tol, tol]
        
    else:
        k = k
        
    if type(k) == list and type(k[1]) == float:
        
        k = [tol, k[1]]
    
    else:
        k = k
        
    if type(k) == list and type(k[1]) == int:
        
        k = [tol, k[1]]
    
    else:
        k = k
    
    
    if type(k) == int:
            
            k = (k, k)
            
    elif type(k) == float: 
            k = (k, k)
            
    else: 
            k = k 
            
    if type(n) == int:
        
            n = (n, n)    
            
    elif type(n) == float: 
            n = (n, n)
            
    else: 
            n = n
            
    if np.size(k) > 1 and k[0] > k[1] or np.size(n) > 1 and n[0] > n[1]:
            
        import sys
            
        sys.exit("Lower Bound cannot be greater than Upper Bound")
        
    else:
           
        Fx = np.linspace(0, 1, nsteps)
            
        dist_left = beta(k[0], n[1]-k[0]+1)
        dist_right = beta(k[1]+1, n[0]-k[1])
            
        param_all = {'param_alpha_left'  : k[0],
                        'param_beta_left'   : n[1]-k[0]+1, 
                         'param_alpha_right' : k[1]+1,
                         'param_beta_right'  : n[0]-k[1]}
            
        if k == n:
            lower_bound = dist_left.cdf(Fx)
            import numpy as np
            upper_bound = np.repeat(1-tol, nsteps)
            flag = True
            
        else:
            lower_bound = dist_left.cdf(Fx)
            upper_bound = dist_right.cdf(Fx)
            flag = False
    
    return {'support' : Fx, 'lb': lower_bound, 'ub': upper_bound, 'betaparam': param_all, 'flag': flag}

def plotcBox(Cbox):
    
    tol = 1e-06
    if Cbox.get('flag') == True:
        from matplotlib import pyplot as plt
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        if type(Cbox.get('support')) != list:
        
            plt.plot(Cbox.get('support'), Cbox.get('lb'), 'r')
            plt.axvline(x = 1-tol, color = 'b')
        else:
            plt.step(Cbox.get('lb'), Cbox.get('support')[0], where ='pre')
            plt.step(Cbox.get('ub'), Cbox.get('support')[1], where ='post')
            
    elif Cbox.get('flag') == False:
        from matplotlib import pyplot as plt
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        if type(Cbox.get('support')) != list:
        
            plt.plot(Cbox.get('support'), Cbox.get('lb'), 'r')
            plt.plot(Cbox.get('support'), Cbox.get('ub'), 'b')
        else:
            
            plt.step(Cbox.get('lb'), Cbox.get('support')[0], where ='pre')
            plt.step(Cbox.get('ub'), Cbox.get('support')[1], where ='post')
            
    elif Cbox.get('flag') == 'unknown':
        from matplotlib import pyplot as plt
        
        if type(Cbox.get('support')) != list:
        
            plt.plot(Cbox.get('support'), Cbox.get('lb'), 'r')
            plt.plot(Cbox.get('support'), Cbox.get('ub'), 'b')
        else:
            
            plt.step(Cbox.get('lb'), Cbox.get('support')[0], where ='pre')
            plt.step(Cbox.get('ub'), Cbox.get('support')[1], where ='post')
                
    return

def addIntervals(interval_a, interval_b):
    lower = interval_a[0] + interval_b[0]
    upper = interval_a[1] + interval_b[1]
    return [lower, upper]

def subtractIntervals(interval_a, interval_b):
    lower = interval_a[0] - interval_b[1]
    upper = interval_a[1] - interval_b[0]
    return [lower, upper]
    
def multiplyIntervals(interval_a, interval_b):
    import numpy as np                                       
    lower = np.min([interval_a[0] * interval_b[0], interval_a[0] * interval_b[1], interval_a[1] * interval_b[0],
                interval_a[1] * interval_b[1]])
    upper = np.max([interval_a[0] * interval_b[0], interval_a[0] * interval_b[1], interval_a[1] * interval_b[0],
                interval_a[1] * interval_b[1]])
    return [lower, upper]

def divideIntervals(interval_a, interval_b):
    if interval_b[0] == 0 or interval_b[1] == 0:    
        Ynum = [-float('inf'), float('inf')]
        out = multiplyIntervals(interval_a, Ynum)
    else:       
        Ynum = [1/interval_b[1], 1/interval_b[0]]
        out = multiplyIntervals(interval_a, Ynum)
    return out

def linear_interpolate(x_values, y_values, x):
    import scipy.interpolate
    y_interp = scipy.interpolate.interp1d(x_values, y_values) 
    return y_interp(x).tolist()

def computeConfidenceInterval(Cbox, alpha_level = 0.05, beta_level = 0.95, show_plot = False):
    
    tol = 1e-6
     
    if Cbox.get('betaparam') != 'unknown' and Cbox.get('flag') == False:
        
        from scipy.stats import beta
        
        parameters = Cbox.get('betaparam')
        left_interval = beta.ppf(alpha_level, parameters['param_alpha_left'], parameters['param_beta_left'])
        right_interval = beta.ppf(beta_level, parameters['param_alpha_right'], parameters['param_beta_right'])
            
        # plot focal elements      
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.plot((0, left_interval), (alpha_level, alpha_level), 'r--')
            plt.plot((left_interval, left_interval), (0, alpha_level), 'r--')
            
            plt.plot((0, right_interval), (beta_level, beta_level), 'b--')
            plt.plot((right_interval, right_interval), (0, beta_level), 'b--')
                        
            # plot original pbox
            x = Cbox.get('support')
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            plt.plot(x, lower_bound)
            plt.plot(x, upper_bound)
            
    elif Cbox.get('betaparam') != 'unknown' and Cbox.get('flag') == True:
        
        from scipy.stats import beta
        
        parameters = Cbox.get('betaparam')
        
        left_interval = beta.ppf(alpha_level, parameters['param_alpha_left'], parameters['param_beta_left'])
        
        right_interval = 1-tol
        
        # plot focal elements      
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.plot((0, left_interval), (alpha_level, alpha_level), 'r--')
            plt.plot((left_interval, left_interval), (0, alpha_level), 'r--')
            
            plt.plot((0, right_interval), (beta_level, beta_level), 'b--')
            plt.plot((right_interval, right_interval), (0, beta_level), 'b--')
                        
            # plot original pbox
            x = Cbox.get('support')
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            plt.plot(x, lower_bound)
            plt.plot(x, upper_bound)
          
    elif Cbox.get('betaparam') == 'unknown' and Cbox.get('flag') == 'unknown':
        
        import numpy as np
        
        if type(Cbox.get('support')) != list:

            left_interval = linear_interpolate(Cbox.get('support').tolist(), Cbox.get('lb').tolist(), alpha_level)
            right_interval = linear_interpolate(Cbox.get('support').tolist(), Cbox.get('ub').tolist(), beta_level)
                            
        elif type(Cbox.get('support')) == list and np.shape(Cbox.get('support'))[0] == 2:
            
            left_interval = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), alpha_level)
            right_interval = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), beta_level)
                                
                        
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.plot((0, left_interval), (alpha_level, alpha_level), 'r--')
            plt.plot((left_interval, left_interval), (0, alpha_level), 'r--')
            
            plt.plot((0, right_interval), (beta_level, beta_level), 'b--')
            plt.plot((right_interval, right_interval), (0, beta_level), 'b--')
                        
            # plot original pbox
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            
            
            plt.step(Cbox.get('lb'), Cbox.get('support')[0], where ='pre')
            plt.step(Cbox.get('ub'), Cbox.get('support')[1], where ='post')
        
    return [left_interval, right_interval]

def cBoxcBox(kCbox, nCbox, npoints = 100):
    
    focal_elements_k = computeFocalElements(kCbox, npoints = npoints)
    focal_elements_n = computeFocalElements(nCbox, npoints = npoints)
    
    Cbox_list = []
    
    for i in range(npoints):
        Cbox = cBox(focal_elements_k[i], focal_elements_n[i])
        Cbox_list.append(Cbox)
        
    lower_values_list = []
    upper_values_list = []

    for j in range(npoints):    
        lower = Cbox_list[j]['lb']
        upper = Cbox_list[j]['ub']
        lower_values_list.append(lower)
        upper_values_list.append(upper)
    
    import numpy as np  
    average_lower = np.sum(lower_values_list, axis  = 0)/npoints
    average_upper = np.sum(upper_values_list, axis  = 0)/npoints
    x = Cbox_list[0]['support'] 
    flag = 'unknown'
    
    return {'support' : x, 'lb': average_lower, 'ub': average_upper, 'betaparam': 'unknown', 'flag': flag} 

def computeFocalElements(Cbox, npoints = 100, show_plot = False):
    
    tol = 1e-6
    import numpy as np

    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
 
    if Cbox.get('betaparam') != 'unknown' and Cbox.get('flag') == False:
        
        from scipy.stats import beta
        
        parameters = Cbox.get('betaparam')
        
        left_focal_elements = beta.ppf(Fx_left, parameters['param_alpha_left'], parameters['param_beta_left'])
        right_focal_elements = beta.ppf(Fx_right, parameters['param_alpha_right'], parameters['param_beta_right'])
        left_focal_elements = left_focal_elements.tolist()
        right_focal_elements = right_focal_elements.tolist() 
        focal_elements = [[lf, rf] for lf, rf in zip(left_focal_elements, right_focal_elements)]
        
        # plot focal elements      
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.scatter(left_focal_elements, Fx_left)
            plt.scatter(right_focal_elements, Fx_right)
                        
            plt.step(left_focal_elements, Fx_left, where = 'pre')
            plt.step(right_focal_elements, Fx_right, where = 'post')
            
            # plot original pbox
            x = Cbox.get('support')
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            plt.plot(x, lower_bound)
            plt.plot(x, upper_bound)
    
    elif Cbox.get('betaparam') != 'unknown' and Cbox.get('flag') == True:
        
        from scipy.stats import beta
        
        parameters = Cbox.get('betaparam')
        
        left_focal_elements = beta.ppf(Fx_left, parameters['param_alpha_left'], parameters['param_beta_left'])
        
        import numpy as np
        right_focal_elements = np.repeat(1-tol, npoints)
        left_focal_elements = left_focal_elements.tolist()
        right_focal_elements = right_focal_elements.tolist() 
        focal_elements = [[lf, rf] for lf, rf in zip(left_focal_elements, right_focal_elements)]
        
        # plot focal elements      
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            plt.scatter(left_focal_elements, Fx_left)
            plt.scatter(right_focal_elements, Fx_right)
                        
            plt.step(left_focal_elements, Fx_left, where = 'pre')
            plt.step(right_focal_elements, Fx_right, where = 'post')
            
            # plot original pbox
            x = Cbox.get('support')
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            plt.plot(x, lower_bound)
            plt.plot(x, upper_bound)
        
    elif Cbox.get('betaparam') == 'unknown' and Cbox.get('flag') == 'unknown':
        
        left_fe_cb1 = []
        right_fe_cb1 = []
        
        # loop over number of samples
        for i in range(npoints):
            
            if np.shape(Cbox.get('support'))[0] == 1:
            # first cbox
                left_focal_elements_cb1 = linear_interpolate(Cbox.get('support').tolist(), Cbox.get('lb').tolist(), Fx_left[i])
                right_focal_elements_cb1 = linear_interpolate(Cbox.get('support').tolist(), Cbox.get('ub').tolist(), Fx_right[i])
                left_fe_cb1.append(left_focal_elements_cb1)
                right_fe_cb1.append(right_focal_elements_cb1)
                
            elif np.shape(Cbox.get('support'))[0] == 2:
                left_focal_elements_cb1 = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_left[i])
                right_focal_elements_cb1 = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_right[i])
                left_fe_cb1.append(left_focal_elements_cb1)
                right_fe_cb1.append(right_focal_elements_cb1)
                
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)]
            
        if show_plot == True:
            
            import matplotlib.pyplot as plt
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.step(left_fe_cb1, Fx_left, where='pre')
            plt.step(right_fe_cb1, Fx_right, where= 'post')
            # plot original pbox
            x = Cbox.get('support')
            lower_bound = Cbox.get('lb')
            upper_bound = Cbox.get('ub')
            plt.plot(x, lower_bound)
            plt.plot(x, upper_bound)
        
    return focal_elements

def addCbox(Cbox1, Cbox2, npoints=100, show_plot = False):   
    
    import numpy as np   
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox1.get('betaparam') =='unknown' and Cbox2.get('betaparam') == 'unknown':
                        
        left_fe_cb1 = []
        right_fe_cb1 = []
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of points
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            # second cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)] 
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
        
    elif Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') != 'unknown':

        left_fe_cb1 = []
        right_fe_cb1 = []
        
        # loop over number of samples
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)]
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
        
    elif Cbox1.get('betaparam') != 'unknown' and Cbox2.get('betaparam') == 'unknown':
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of samples
        for i in range(npoints):            
            # first cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left)
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right)
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
      
    else:
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)    
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
    
    # preallocation
    out = np.zeros((len(focal_element_Cbox2)**2, 2)) 
     
    for j in range(len(focal_element_Cbox1)):
        
        for k in range(len(focal_element_Cbox2)):
            
            out[j + k * len(focal_element_Cbox1)] = addIntervals(focal_element_Cbox1[j],
                     focal_element_Cbox2[k])
    # sort out in increasing order       
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    # preallocation
    lower_bound = np.zeros(len(focal_element_Cbox1))
    upper_bound = np.zeros(len(focal_element_Cbox1))
    
    for l in range(len(focal_element_Cbox1)):
        lower_bound[l] = lower_sort[l * len(focal_element_Cbox1) + len(focal_element_Cbox1)-1]
        upper_bound[l] = upper_sort[l*len(focal_element_Cbox1)]
        
    if show_plot == True:
            
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}

def subtractCbox(Cbox1, Cbox2, npoints = 100, show_plot = False):
    
    import numpy as np   
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') == 'unknown':
                        
        left_fe_cb1 = []
        right_fe_cb1 = []
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of points
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            # second cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)] 
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
        
    elif Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') != 'unknown':

        left_fe_cb1 = []
        right_fe_cb1 = []
        
        # loop over number of points
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)]
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
        
    elif Cbox1.get('betaparam') != 'unknown' and Cbox2.get('betaparam') == 'unknown':
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of samples
        for i in range(npoints):            
            # first cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left)
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right)
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
      
    else:
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)    
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
    
    # preallocation
    out = np.zeros((len(focal_element_Cbox2)**2, 2)) 
     
    for j in range(len(focal_element_Cbox1)):
        
        for k in range(len(focal_element_Cbox2)):
            
            out[j + k * len(focal_element_Cbox1)] = subtractIntervals(focal_element_Cbox1[j],
                     focal_element_Cbox2[k])
            
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    # preallocation
    lower_bound = np.zeros(len(focal_element_Cbox1))
    upper_bound = np.zeros(len(focal_element_Cbox1))
    
    for l in range(len(focal_element_Cbox1)):
        lower_bound[l] = lower_sort[l * len(focal_element_Cbox1) + len(focal_element_Cbox1)-1]
        upper_bound[l] = upper_sort[l*len(focal_element_Cbox1)]
            
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}

def multiplyCbox(Cbox1, Cbox2, npoints = 100, show_plot = False):
    
    import numpy as np   
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') == 'unknown':
                        
        left_fe_cb1 = []
        right_fe_cb1 = []
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of points
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            # second cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)] 
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
        
    elif Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') != 'unknown':

        left_fe_cb1 = []
        right_fe_cb1 = []
        
        # loop over number of samples
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)]
        
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
        
    elif Cbox1.get('betaparam') != 'unknown' and Cbox2.get('betaparam') == 'unknown':
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of samples
        for i in range(npoints):            
            # first cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left)
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right)
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)
        
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
      
    else:
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)    
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
    
    # preallocation
    out = np.zeros((len(focal_element_Cbox2)**2, 2)) 
     
    for j in range(len(focal_element_Cbox1)):
        
        for k in range(len(focal_element_Cbox2)):
            
            out[j + k * len(focal_element_Cbox1)] = multiplyIntervals(focal_element_Cbox1[j],
                     focal_element_Cbox2[k])
            
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    # preallocation
    lower_bound = np.zeros(len(focal_element_Cbox1))
    upper_bound = np.zeros(len(focal_element_Cbox1))
    
    for l in range(len(focal_element_Cbox1)):
        lower_bound[l] = lower_sort[l * len(focal_element_Cbox1) + len(focal_element_Cbox1)-1]
        upper_bound[l] = upper_sort[l*len(focal_element_Cbox1)]
    
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}

def divideCbox(Cbox1, Cbox2, npoints = 100, show_plot = False):
    
    import numpy as np   
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') == 'unknown':
                        
        left_fe_cb1 = []
        right_fe_cb1 = []
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of points
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            # second cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)] 
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
        
    elif Cbox1.get('betaparam') == 'unknown' and Cbox2.get('betaparam') != 'unknown':

        left_fe_cb1 = []
        right_fe_cb1 = []
        
        # loop over number of samples
        for i in range(npoints):
            # first cbox
            left_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[0].tolist(), Cbox1.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements_cb1 = linear_interpolate(Cbox1.get('support')[1].tolist(), Cbox1.get('ub').tolist(), Fx_list_right[i])
            left_fe_cb1.append(left_focal_elements_cb1)
            right_fe_cb1.append(right_focal_elements_cb1)
            
        focal_element_Cbox1 = [[lf, rf] for lf, rf in zip(left_fe_cb1, right_fe_cb1)]
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
        
    elif Cbox1.get('betaparam') != 'unknown' and Cbox2.get('betaparam') == 'unknown':
        
        left_fe_cb2 = []
        right_fe_cb2 = []
        
        # loop over number of samples
        for i in range(npoints):            
            # first cbox
            left_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[0].tolist(), Cbox2.get('lb').tolist(), Fx_list_left)
            right_focal_elements_cb2 = linear_interpolate(Cbox2.get('support')[1].tolist(), Cbox2.get('ub').tolist(), Fx_list_right)
            left_fe_cb2.append(left_focal_elements_cb2)
            right_fe_cb2.append(right_focal_elements_cb2)
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)
        focal_element_Cbox2 = [[lf, rf] for lf, rf in zip(left_fe_cb2, right_fe_cb2)]
      
    else:
        
        focal_element_Cbox1 = computeFocalElements(Cbox1, npoints=npoints)    
        focal_element_Cbox2 = computeFocalElements(Cbox2, npoints=npoints)
    
    # preallocation
    out = np.zeros((len(focal_element_Cbox2)**2, 2)) 
     
    for j in range(len(focal_element_Cbox1)):
        
        for k in range(len(focal_element_Cbox2)):
            
            out[j + k * len(focal_element_Cbox1)] = divideIntervals(focal_element_Cbox1[j],
                     focal_element_Cbox2[k])
            
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    # preallocation
    lower_bound = np.zeros(len(focal_element_Cbox1))
    upper_bound = np.zeros(len(focal_element_Cbox1))
    
    for l in range(len(focal_element_Cbox1)):
        lower_bound[l] = lower_sort[l * len(focal_element_Cbox1) + len(focal_element_Cbox1)-1]
        upper_bound[l] = upper_sort[l*len(focal_element_Cbox1)]
    
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}

def addCboxNnumber(Cbox, num, npoints = 100, show_plot = False):
        
    import numpy as np   
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()

    if Cbox.get('betaparam') == 'unknown':
        
        left_fe = []
        right_fe = []
        
        for i in range(npoints):
            
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            left_fe.append(left_focal_elements)
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    else:
        
        focal_elements = computeFocalElements(Cbox, npoints=npoints)
        
    if type(num) == int:
        
        num = [num, num]
        
    else:
        
        num = num
    
    if type(num) == float:
        
        num = [num, num]
        
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
    
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = addIntervals(focal_elements[j],
                        focal_element_num[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))
   
    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
        
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
        
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}

def subtractCboxNnumber(Cbox, num, npoints = 100, show_plot = False):
    
    import numpy as np
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox.get('betaparam') == 'unknown':
        
        left_fe = []
        right_fe = []
        
        for i in range(npoints):
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            left_fe.append(left_focal_elements)
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    elif Cbox.get('betaparam') != 'unknown' or '' :
        
        focal_elements = computeFocalElements(Cbox, npoints = npoints)
        
    if type(num) == int:
        
        num = [num, num]
        
    else:
        
        num = num
        
    if type(num) == float:
        
        num = [num, num]
    
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
            
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = subtractIntervals(focal_elements[j],
                        focal_element_num[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))
    
    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
        
    if show_plot == True:
    
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}
    
def multiplyCboxNnumber(Cbox, num, npoints = 100, show_plot = False):
    
    import numpy as np
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox.get('betaparam') == 'unknown':
        
        left_fe = []
        right_fe = []
        
        for i in range(npoints):
            
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            left_fe.append(left_focal_elements)
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    elif Cbox.get('betaparam') != 'unknown':
        
        focal_elements = computeFocalElements(Cbox, npoints = npoints)
        
    if type(num) == int:
        
        num = [num, num]
        
    else:
        
        num = num
        
    if type(num) == float:
        
        num = [num, num]
        
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
            
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = multiplyIntervals(focal_elements[j],
                        focal_element_num[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))
   
    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
    
    if show_plot == True:     
        
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}
    
def divideCboxNnumber(Cbox, num, npoints = 100, show_plot = False):
    
    import numpy as np
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox.get('betaparam') == 'unknown':     
        
        left_fe = []        
        right_fe = []
        
        for i in range(npoints):  
            
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])         
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            left_fe.append(left_focal_elements)           
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    elif Cbox.get('betaparam') != 'unknown':
        
        focal_elements = computeFocalElements(Cbox, npoints = npoints)
        
    if type(num) == int:
        
        num = [num, num]
        
    else:
        
        num = num
        
    if type(num) == float:
        
        num = [num, num]
    
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
            
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = divideIntervals(focal_elements[j],
                        focal_element_num[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))

    
    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
    
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}
        
def subtractNumberNcbox(num, Cbox, npoints = 100, show_plot = False):
    
    import numpy as np
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox.get('betaparam') == 'unknown':
        
        left_fe = []
        right_fe = []
        
        for i in range(npoints):
            
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            left_fe.append(left_focal_elements)
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    elif Cbox.get('betaparam') != 'unknown':
        
        focal_elements = computeFocalElements(Cbox, npoints = npoints)
        
    if type(num) == int:
        
        num = [num, num]      
    else:  
        
        num = num
        
    if type(num) == float:
        
        num = [num, num]
    
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
        
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = subtractIntervals(focal_element_num[j],
                        focal_elements[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))
    
    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
        
    if show_plot == True:
            
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
        
    flag = 'unknown' 
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}


def divideNumberNcbox(num, Cbox, npoints = 100, show_plot = False):
    
    import numpy as np
    
    Fx_left = np.linspace(0, (npoints-1)/npoints, npoints)
    Fx_right = np.linspace(1, npoints, npoints)/npoints
            
    Fx_list_left = Fx_left.tolist()
    Fx_list_right = Fx_right.tolist()
    
    if Cbox.get('betaparam') == 'unknown':
        
        left_fe = []
        
        right_fe = []
        
        for i in range(npoints):
                        
            left_focal_elements = linear_interpolate(Cbox.get('support')[0].tolist(), Cbox.get('lb').tolist(), Fx_list_left[i])            
            right_focal_elements = linear_interpolate(Cbox.get('support')[1].tolist(), Cbox.get('ub').tolist(), Fx_list_right[i])
            
            left_fe.append(left_focal_elements)
            
            right_fe.append(right_focal_elements)
            
        focal_elements = [[lf, rf] for lf, rf in zip(left_fe, right_fe)]
    
    elif Cbox.get('betaparam') != 'unknown':
        
        focal_elements = computeFocalElements(Cbox, npoints = npoints) 
        
    if type(num) == int:   
        
        num = [num, num]       
    else:      
        
        num = num 
        
    if type(num) == float:
        
        num = [num, num]
    
    else:
        
        num = num
        
    focal_element_num = [num]*(npoints)
    
    out = np.zeros((len(focal_elements)**2, 2)) 
     
    for j in range(len(focal_elements)):
        
        for k in range(len(focal_element_num)):
            
            out[j + k * len(focal_elements)] = divideIntervals(focal_element_num[j],
                        focal_elements[k])
                
    sort_out = np.sort(out, 0)
    lower_sort = sort_out[:,0]
    upper_sort = sort_out[:,1]
    
    lower_bound = np.zeros(len(focal_elements))
    upper_bound = np.zeros(len(focal_elements))

    for l in range(len(focal_elements)):
        lower_bound[l] = lower_sort[l * len(focal_elements) + len(focal_elements)-1]
        upper_bound[l] = upper_sort[l*len(focal_elements)]
        
    if show_plot == True:
        
        import matplotlib.pyplot as plt
        
        plt.step(lower_bound, Fx_left, where ='pre')
        plt.step(upper_bound, Fx_right, where ='post')
    
    flag = 'unknown'
     
    return {'support' : [Fx_left, Fx_right], 'lb': lower_bound, 'ub': upper_bound, 'betaparam': 'unknown', 'flag': flag}


        







    

