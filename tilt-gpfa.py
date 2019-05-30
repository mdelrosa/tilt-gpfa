# tilt-gpfa.py
# perform gpfa on tilt dataset

import funs.util as util
import funs.datamanager as dm
import funs.engine as engine
import matplotlib.pyplot as plt
import numpy as np
import json
import dill
import sys

# use to save previous jupyter envs
# -> fitOnline_5.db => used fitOnline with batchSize=5
# dill.dump_session('fitBatch5_correctTrialsAndEst.db')

# load previous jupyter envs
# dill.load_session('fitBatch5_correctTrialsAndEst.db')

def load_data(binSize,binFilename,classifierFilename):
    # load binned resp data
    with open(binFilename) as json_file:
        data=json.load(json_file)

    # load classified trial indices
    with open(classifierFilename) as json_file:
        data_classifier=json.load(json_file)
    
    # get correct trials for given binSize
    binStr='bin_{}ms'.format(binSize)
    correct_trials = data_classifier[binStr]['correct_trials']
    return [correct_trials, data]

def iter_gpfa(data_in,latent_dims,params,fig_params):
    # -> iterate through different dimensions for latent space
    # unpack params
    ydim      = params['ydim']
    trialDur  = params['trialDur']
    binSize   = params['binSize']
    maxEMiter = params['maxEMiter']
    dOffset   = params['dOffset']
    infType   = params['infType']
    numTrials = params['numTrials']
    
    # unpack fig_params
    preamble  = fig_params['preamble']

    # generate tilt dataset obj
    tilt_set = dm.TiltDataset(
                                data = data_in,
                                ydim = ydim,
                                trialDur = trialDur,
                                binSize = binSize,
                                numTrials = numTrials,
                                numTrData = True
                            )
    # iterate through # of latent dimensions
    out = dict()
    for xdim in latent_dims:
        print('-> Running gpfa for xdim={}'.format(xdim))
        initParams = util.initializeParams(xdim, ydim, tilt_set)
        if infType=='batch':
            # Fit using vanilla (batch) EM; longer runtime
            fitBatch = engine.PPGPFAfit(
                experiment 		= tilt_set,
                initParams 		= initParams,
                inferenceMethod = 'laplace',
                EMmode 			= 'Batch',
                maxEMiter 		= maxEMiter)
            out[xdim]=fitBatch
        elif infType=='online':
            # Fit using online EM; shorter runtime
            fitOnline = engine.PPGPFAfit(
                experiment 		= tilt_set,
                initParams 		= initParams,
                EMmode 			= 'Online',
                maxEMiter 		= maxEMiter,
                inferenceMethod = 'laplace',
                batchSize 		= 5)
            # plot, save figs
            fitOnline.plotParamSeq();
            plt.savefig('{}-paramSeq-xdim{}.png'.format(preamble,xdim))
            plt.show()
            fitOnline.plotTrajectory(1);
            plt.savefig('{}-latentTraj-xdim{}.png'.format(preamble,xdim))
            plt.show()
            out[xdim]=fitOnline
    return out
        
def process_tilt_data(binSize,correct_trials,data,latent_dims):
    # -> data = binned json data to be formatted for gpfa
    # -> latent_dims = array of ints for different latent trajectory sizes

    fitObj = dict()
    i_trial = 0
    for event_name, event_data in data[str(binSize)]['data'].items():
        print('event: {}'.format(event_name))
        data_in=[]
        for trial_name, trial_data in event_data.items():
            if i_trial in correct_trials:
                data_in.append({'Y':np.asarray(trial_data['Y'])})
            i_trial = i_trial + 1
        numTrials = len(data_in)
        data_sub=data['5']
        # Specify dataset & fitting parameters
        ydim      = data_in[0]['Y'][:,1].size # neurons		
        trialDur  = data_sub['trialDur'] # in ms
        binSize   = data_sub['binSize']	 # in ms
        maxEMiter = 100	# expectation maximization iterations	
        dOffset   = 1	 # controls firing rate

        params={
                'ydim':ydim,
                'trialDur':trialDur,
                'binSize':binSize,
                'maxEMiter':maxEMiter,
                'dOffset':dOffset,
                'infType':'batch',
                'numTrials': numTrials
               }

        filepath = "images/"
        trial_type = "allTrials"
        preamble = "{}{}-{}".format(filepath,event_name,trial_type)

        fig_params={
                    'preamble': preamble
                    }

        fitObj[event_name]=iter_gpfa(data_in,latent_dims,params,fig_params)

def plot_3d_traj(filepath,trial_type,preamble):
    # plot multiple latent trajectories
    filepath = "images/"
    trial_type = "fitBatch5_correctTrialsAndEst"
    preamble = "{}{}".format(filepath,trial_type)
    # trial_i = 0
    for event_name, event_data in fitObj.items():
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca(projection='3d')
        fitObjTemp=event_data[3]
        print('event: {}'.format(event_name))
    #     print(len(fitObjTemp.infRes['post_mean']))
    #     print(fitObjTemp.infRes['post_mean'])
        for traj in fitObjTemp.infRes['post_mean']:
    #         traj = fitObj.infRes['post_mean']
    #         if trial_i in correct_trials:
            ax.plot3D(traj[0],traj[1],traj[2])
    #         trial_i = trial_i + 1
        plt.title('{} Latent Trajectory'.format(event_name));

        ax.set_xlabel('xdim1')
        ax.set_ylabel('xdim2')
        ax.set_zlabel('xdim3')
    #     plt.show();
        plt.savefig('{}-{}-latentTraj-3D.png'.format(preamble,event_name))

def plot_trial_traj(filepath,correct_trials,dim,fitObj):
    # filepath = 'C:/Users/Mason/Box Sync/UC Davis/00 Quarters/Spring 2019/MAE298/03 Project/poisson-gpfa/images/classifiedTrialsAndEst/latentTraj'
    incr=1
    # dim=3
    last_i = 0
    for event_name, event_data in fitObj.items():
        num_trials = len(event_data[dim].infRes['post_mean'])
        print('num_trials: {}'.format(num_trials))
        for i in range(0,numTrials+1):
            trial_num = correct_trials[last_i+i]
    #         print("plotting trial #{:03d}".format(trial_num))
            title_str = 'Trial #{:03d}'.format(trial_num)
            save_str = filepath+'\{}-trial{:03d}-{}traj'.format(event_name,trial_num,dim)
            fitObj[event_name][dim].plotTrajectory(i);plt.title(title_str);plt.savefig(save_str);
    #         plt.show();
            plt.close();
        last_i = last_i + num_trials
        print('last_i: {}'.format(last_i))

if __name__ == '__main__':
    # sys args
    n_args = len(sys.argv)
    if n_args == 2:
        min_dim = 3
        max_dim = int(sys.argv[1])
        incr    = 2
    elif n_args == 3:
        min_dim = int(sys.argv[1])
        max_dim = int(sys.argv[2])
        incr    = 2
    elif n_args == 4:
        min_dim = int(sys.argv[1])
        max_dim = int(sys.argv[2])
        incr    = int(sys.argv[3])
    else:
        # wrong number of sys args; use predefined range
        min_dim = 3
        max_dim = 11
        incr    = 2
    latent_dims = list(range(min_dim,max_dim,incr))
    print("Latent Dimensions to infer: {}".format(latent_dims))
    # load data
    print("Loading data...")
    binSize=5
    binFilename='C:/Users/Mason/Box Sync/UC Davis/00 Quarters/Spring 2019/MAE298/03 Project/data/gpfa_inputs.json'
    classifierFilename='C:/Users/Mason/Box Sync/UC Davis/00 Quarters/Spring 2019/MAE298/03 Project/data/tilt_psthclassifier.json'
    [correct_trials, data]=load_data(binSize,binFilename,classifierFilename)
    # perform gpfa for different latent trajectory dimensionalities
    print("Begin gpfa iteration...")
    fitObj=process_tilt_data(binSize,correct_trials,data,latent_dims)
    # dump data for local post-process
    dill.dump_session('fitBatch_{}to{}by{}.db'.format(min_dim,max_dim,incr))