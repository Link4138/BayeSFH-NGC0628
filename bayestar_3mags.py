from numpy import append, array, exp, linspace, loadtxt, log10, pi, meshgrid, savetxt, sqrt, where, ones, percentile
from scipy import special
import matplotlib.pyplot as plt
import time, glob, os, sys
import multiprocessing as mp
import argparse
import stan
import pickle



if sys.version_info[0] < 3:
    raise Exception ("Must be using Python 3")
    sys.exit()


## Trapezoidal integration (one dimention) 
def trapz(yt,xt):
    del_x = xt[1:]-xt[:len(xt)-1]
    y2 = 0.5*(yt[1:]+yt[:len(yt)-1])
    return sum(y2*del_x)




#####################################################################################
#####################################################################################
################################ POSTERIOR COMPUT ###################################
#####################################################################################
#####################################################################################



################################# LIKELIHOOD ###################################

def Normal_MGk(fw_dat,fw_err,Iso_sig):                  ## Error model apparent maginute ##
    sig2 = fw_err*fw_err+Iso_sig*Iso_sig                ##     and normal Isochrone      ##
    return  lambda fw_iso : exp( -0.5*(fw_dat-fw_iso)**2 / sig2 ) / sqrt(2.*pi*sig2)


def Phi_MGk(fwj2, sig_fwj2, fwklim, sig_i2):
    b = sig_i2*sig_i2+sig_fwj2*sig_fwj2
    b1 = sig_i2*sig_i2/b
    b2 = sig_fwj2*sig_fwj2/b
    b3 = sig_i2*sig_fwj2/sqrt(b)
    return  lambda fw_i2 : special.ndtr( ( fwklim - b1*fwj2 - b2*fw_i2 ) / b3 )



################################### PRIORS #####################################

def IMF_Krp(m, ml=0.1, mint=0.5, mu=350.,a1=1.3,a2=2.3):

    h2 = (mu**(1.-a2)-mint**(1.-a2))/(1.-a2)
    h1 = (mint**(1.-a1)-ml**(1.-a1))/(1.-a1)

    c1 = 1./(h1+h2*mint**(a2-a1))
    c2 = c1*mint**(a2-a1)

    c = ones(len(m))
    c[where(m < mint)] = c1
    c[where(m >= mint)] = c2

    a = ones(len(m))
    a[where(m < mint)] = -a1
    a[where(m >= mint)] = -a2
    imf = c*m**a

    return(imf)


#def IMF_Salp(m):
#    return 




################################## POSTERIOR ###################################


#    0      1       2           3           4           5           6           7
#    RA    DEC     fw1      fw1_error      fw2      fw2_error      fw3      fw3_error
def P_ij_map(IDp, dat, Ndat, Iso, Niso, fw1_lim=20., fw2_lim=20., fw3_lim=20., sig_i=0.1, imf="Krp"):
    filename_p = '%s_Pij_Data_LimMag%.2lf_%srows_%siso_IsoModel_sig%s_IMF_%s_Simple.txt' % (IDp,fw2_lim,str(Ndat),str(Niso),str(sig_i).replace('.','p'), imf)    ## Opening file
    fp = open(os.path.join("pij_cij_results",filename_p),'a')
    args = []

    for j in range(Ndat):                        ## Pij is calcutated row by row, i.e. fix j-th dat and run each i-th isochrone.
#                    0   1     2    3     4         5        6      7       8        9     10
        args.append([j, dat, Niso, Iso, fw1_lim, fw2_lim, fw3_lim, Ndat, filename_p, sig_i, imf])

    with mp.Pool(mp.cpu_count()-1) as p:         ## Pooling Pij rows using all the abailable CPUs (Parallel computation)
        results = p.map(P_ij_row_map, args)
        Pij_out=[]
        for [j,wr] in results:
            fp.write('{}'.format(' '.join(wr))+'\n')
            Pij_out.append(array(wr, dtype=float))
    fp.close()
    
    return([Pij_out, filename_p])



def P_ij_row_map(args):
    j = args[0]
    dat = args[1]
    Niso = args[2]
    Iso = args[3]

    fw1_lim = args[4]
    fw2_lim = args[5]
    fw3_lim = args[6]

    Ndat = args[7]
    filename_p = args[8]
    sig_i = args[9]
    imf = args[10]

    P_fw1 = Normal_MGk(dat[2][j],dat[3][j],sig_i)
    P_fw2 = Normal_MGk(dat[4][j],dat[5][j],sig_i)
    P_fw3 = Normal_MGk(dat[6][j],dat[7][j],sig_i)

    Phi_fw1 = Phi_MGk(dat[2][j], dat[3][j], fw1_lim, sig_i)
    Phi_fw2 = Phi_MGk(dat[4][j], dat[5][j], fw2_lim, sig_i)
    Phi_fw3 = Phi_MGk(dat[6][j], dat[7][j], fw3_lim, sig_i)

    wr=[]
    for i in range(Niso):                    ## Isochrone loop

        if imf == "Krp":
            imf_p = IMF_Krp(Iso[i][1])
        elif imf == "Slp":
            imf_p = IMF_Salp(Iso[i][1])
        else:
            imf_p = IMF_Krp(Iso[i][1])

        Intg = imf_p*P_fw1(Iso[i][2])*P_fw2(Iso[i][3])*P_fw3(Iso[i][4])*Phi_fw1(Iso[i][2])*Phi_fw2(Iso[i][3])*Phi_fw3(Iso[i][4])

        ## Interand
        p = trapz(Intg,Iso[i][1])

        wr.append(str(p))

    return ([j,wr])




################################ NORMALIZATION CONSTANT ###################################


def C_ij_map(IDc, dat, Ndat, Iso, Niso, fw1_lim=20., fw2_lim=20., fw3_lim=20., sig_i=0.1, imf="Krp"):
    filename_c = '%s_Cij_Data_LimMag%.2lf_%srows_%siso_IsoModel_sig%s_IMF_%s_Simple.txt' % (IDc,fw2_lim,str(Ndat),str(Niso),str(sig_i).replace('.','p'), imf)
    fp = open(os.path.join("pij_cij_results",filename_c),'a')   ## output matrix
    args = []
    for j in range(Ndat):                        ## Cij is calcutated row by row, i.e. fix j-th dat and run each i-th isochrone.
        args.append([j, dat, Niso, Iso, fw1_lim, fw2_lim, fw3_lim, Ndat, filename_c, sig_i, imf])

    with mp.Pool(mp.cpu_count()-1) as p:
        results = p.map(C_ij_row_map, args)
        Cij_out=[]
        for [j,wr] in results:
            fp.write('{}'.format(' '.join(wr))+'\n')
            Cij_out.append(array(wr, dtype=float))
    fp.close()
    
    return(Cij_out)



def C_ij_row_map(args):

    j = args[0]
    dat = args[1]
    Niso = args[2]
    Iso = args[3]

    fw1_lim = args[4]
    fw2_lim = args[5]
    fw3_lim = args[6]

    Ndat = args[7]
    filename_c = args[8]
    sig_i = args[9]
    imf = args[10]

    phi_fw1 = Phi_MGk(dat[2][j], dat[3][j], fw1_lim, sig_i)
    phi_fw2 = Phi_MGk(dat[4][j], dat[5][j], fw2_lim, sig_i)
    phi_fw3 = Phi_MGk(dat[6][j], dat[7][j], fw2_lim, sig_i)

    wr = []
    for i in range(Niso):

        if imf == "Krp":
            imf_c = IMF_Krp(Iso[i][1])
        elif imf=="Slp":
            imf_c = IMF_Salp(Iso[i][1])
        else:
            imf_c = IMF_Krp(Iso[i][1])

        intg_c = imf_c*phi_fw1(Iso[i][2])*phi_fw2(Iso[i][3])*phi_fw3(Iso[i][4])
        p_c = trapz(intg_c,Iso[i][1])

        wr.append(str(p_c))

    return ([j,wr])










#####################################################################################
#####################################################################################
############################### POSTERIOR SAMPLING ##################################
#####################################################################################
#####################################################################################



############ Stan code ############
code = """

functions{
    real P(int N1, int N2, vector v, matrix M) {
        vector[N1] Mj;
        vector[N1] ln_Mj;

        Mj= M*v;
        for (j in 1:N1){
            if (Mj[j]<=0.)
                Mj[j] = 1.;
        }
        ln_Mj = log(Mj);
        return sum(ln_Mj);
    }
}

data {
    int<lower=0> Nj; // number of data
    int<lower=0> Ni; // number of isochrones
    matrix[Nj,Ni] Pij; // Probability matrix
    matrix[Nj,Ni] Cij; // Normalization matrix
}

parameters {
    simplex[Ni] a;
}

model {
    target += dirichlet_lpdf(a | rep_vector(1., Ni));
    target += P(Nj,Ni,a,Pij);
    target += -1.*P(Nj,Ni,a,Cij);
}

"""



############ Sampling routine ############

def ai_samp( pij, cij, Ndat, Niso, Nwlk, Nsmp, ID, Name, Z_age):

    ### Data for STAN ###
    dats = {'Nj' : Ndat,
            'Ni' : Niso,
            'Pij': pij,
            'Cij': cij  }

    ############ Running pystan ############

    sm = stan.build(code, data=dats, random_seed=1234)
    fit = sm.sample(num_samples=Nsmp, num_chains=Nwlk, num_warmup=200)
    a_sp = fit["a"].T

    ######### Saving the MCMC sample #########

    N_iso = len(a_sp[0])

    a_perc = array([ percentile(ai,[10,50,90]) for ai in a_sp.T])       ##  10th, 50th, 90th percentiles

    sfh=array([Z_age[:,0], Z_age[:,1], a_perc[:,0], a_perc[:,1], a_perc[:,2] ]).T

    ##
    hd='       Z       Log_age        p10        p50       p90'
    savetxt(ID+"_ai"+Name+"_Niter"+str(len(a_sp))+".txt", sfh, header=hd, fmt="%.6f", delimiter="  ")





####################################################################################################
###################################### Execution Routines ##########################################
####################################################################################################



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bayessian Methods')
    parser.add_argument("--p", dest='parallel', default=True, action='store_false', help='Unactivate parallel mode')
    parser.add_argument("--step", dest='step', default=1, type=int, help='Step')
    parser.add_argument("--mag1lim", dest='mag1_lim', default=30.0, type=float, help='Upper limit in Apparent fw1 Magnitude')
    parser.add_argument("--mag2lim", dest='mag2_lim', default=30.0, type=float, help='Upper limit in Apparent fw2 Magnitude')
    parser.add_argument("--mag3lim", dest='mag3_lim', default=30.0, type=float, help='Upper limit in Apparent fw3 Magnitude')
    parser.add_argument("--sigmag1", dest='sig_mag1', default=0.1, type=float, help='Isochrone toletance in mag1.')
    parser.add_argument("--sigmag2", dest='sig_mag2', default=0.1, type=float, help='Isochrone toletance in mag2.')
    parser.add_argument("--sigmag3", dest='sig_mag3', default=0.1, type=float, help='Isochrone toletance in mag3.')
    parser.add_argument("--dismod", dest='dm', default=29.67, type=float, help='Distance modulus')
    parser.add_argument("--phase", dest='phase', default=100, type=int, help='Star Phase')
    parser.add_argument("--isofiles", dest='isofiles', default='', type=str, help='Isochrone Files (Path relative to bayestar.py)')
    parser.add_argument("--data", dest='data', default='test_files/data.test', type=str, help='Isochrone Files (Path relative to bayestar.py)')
    parser.add_argument("--folder", dest='folder', default='test_files/Isochrone.test', type=str, help='Isochrone Folder')
    parser.add_argument("--Amag1", dest='A_mag1', default=0.0, type=float, help='Amag1 extinction')
    parser.add_argument("--Amag2", dest='A_mag2', default=0.0, type=float, help='Amag2 extinction')
    parser.add_argument("--Amag3", dest='A_mag3', default=0.0, type=float, help='Amag3 extinction')
    parser.add_argument("--imf", dest='imf', default="Krp", type=str, help='IMF Function (Krp, Slp)')
    parser.add_argument("--walkers", dest='wlk', default=20, type=int, help='Number of walkers for the sampling process.')
    parser.add_argument("--samples", dest='samp', default=500, type=int, help='Number of samplings per walker.')
    options = parser.parse_args()

    N_wlk, N_smp = options.wlk, options.samp

    A_fw1, A_fw2, A_fw3 = options.A_mag1, options.A_mag2, options.A_mag3

    fw1_lim, sig_fw1 = options.mag1_lim, options.sig_mag1                ## Apparent limit magnitus and sigma isochrone model

    fw2_lim, sig_fw2 = options.mag2_lim, options.sig_mag2

    fw3_lim, sig_fw3 = options.mag3_lim, options.sig_mag3

    fk_lim = array([fw1_lim,fw2_lim,fw3_lim])
    sig_fk = array([sig_fw1,sig_fw2,sig_fw3])
    
    dismod = options.dm                                                  ##  Distance modulus

    ########### Reading Isochrones ###########
    if (options.isofiles != ''):
        filelist = []
        f = open(options.isofiles, "r")
        lines = f.readlines()
        for line in lines:
            filelist.append(line.replace('\n',''))

        print (filelist)
        f.close()
    else:
        isodir = options.folder
        filelist = glob.glob(os.path.join(isodir, "**.*"))

    #              F435W  F555W  F814W
    #   ph   mass   mag1   mag2   mag3  Z  log_age
    iso = array([ loadtxt(k) for k in sorted(filelist) ], dtype=object)
    N_iso = len(iso)

    Z_age_isos = array([ loadtxt(k)[0][-2:] for k in sorted(filelist) ], dtype=object)

    ph_sup = options.phase
    m_inf=0.1

    for l in range(N_iso):
        iso[l] = iso[l][where( (iso[l].T[1]>=m_inf) & (iso[l].T[0]<=ph_sup))]       ## mass Truncation & stellar phase Truc
        iso[l] = iso[l].T



    ################################### DATA #######################################

    #  0      1       2        3         4          5         6          7
    #  RA    DEC     fw1   fw1_error    fw2     fw2_error    fw3     fw3_error
    dt = options.data
    step = int(options.step)
    dat = loadtxt(dt)
    msg = "from %d... (FW2 <= %.2lf)" % (len(dat),fw2_lim)
    dat = dat[where(dat[:,4]  < fw2_lim)]        # Truncate by apparent magnitude

    w_dat = dat[:,0]

    # Adding Extinction
    dat[:,2] -= A_fw1+dismod
    dat[:,4] -= A_fw2+dismod
    dat[:,6] -= A_fw3+dismod

    print ("Selecting %d %s" % (len(dat), msg))
    dat = dat[::step]
    dat = dat
    dat = dat.T

    N_dat = len(dat[0])


    ########################### Execution Routines #################################

    start = time.time()
    print ("Starting Pij, Cij computation")
    if (options.parallel):
        print ("\tParallel mode...")

        ID = str(int(time.time()))

        Pij_reslt = P_ij_map(ID, dat, N_dat, iso, N_iso, fw1_lim, fw2_lim, fw3_lim, sig_fw1, options.imf)
        P_ij, Pij_name = Pij_reslt[0], Pij_reslt[1]
        
        Name=Pij_name[Pij_name.find("_Pij")+4:Pij_name.find(".txt")]

        C_ij = C_ij_map(ID, dat, N_dat, iso, N_iso, fw1_lim, fw2_lim, fw3_lim, sig_fw1, options.imf)

    else:
        print ("\tSequential mode...")       ## Not available for the moment
        P_ij(dat, N_dat, r_int, iso, N_iso)
    print ("Finished                                ")


    end = time.time()

    elapsed = end - start

    print ("Elapsed time: %02d:%02d:%02d" % (int(elapsed / 3600.), int((elapsed % 3600)/ 60.), elapsed % 60))


    ###########################################
    ai_samp(P_ij, C_ij, N_dat, N_iso, N_wlk, N_smp, ID, Name, Z_age_isos)



