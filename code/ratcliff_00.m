%% Weiner model v3 
% v1 implements the weiner model with guessing contaminant variable
% v2 keeps trial order intact rather than splitting by condition then looping
% v3 implements delayed response trials
% v4 includes both groups, parameter comparison variables added
% v5 includes extra delta paameters for easy analysis

%% final variables (calculated):

%% final variables (bayesian):
clear;
close all;
addpath("/tmp/yassamri2/MDTO-Diffusion/young_vs_old/trinity-master")
trinity install

proj_id = 'model_00'
model = 'models/model_00.jags';
fid = fopen('models/model_00.params','r');
monitorParameters = textscan(fid,'%s','delimiter','\n'); 
monitorParameters = transpose(monitorParameters{1})
fclose(fid);

load("../data/processed_data/P1_diffusion_trials.mat");

rtP1 = rt;
subjListP1 = subjList;
condListP1 = subList;
load("../data/processed_data/P2_diffusion_trials.mat");
rtP2 = rt;
subjListP2 = subjList;
condListP2 = subList;

% Get subjects and groups 
group_list = [];
subj_group_list = csvread("../data/subj_list.csv", 1,0);
for subj = subjList
    group1 = find(ismember(subj_group_list, [subj,1],'rows'));
    group2 = find(ismember(subj_group_list, [subj,2], 'rows'));
    if isnumeric(group1)
        group_list = [group_list 1];
    elseif isnumeric(group2)
        group_list = [group_list 2];
    else
        disp("No group found for a subject!")
        disp(subj)
        return
    end
end

% Combine them into 1 matrix
y = cat(3, rtP1, rtP2);
y = permute(y, [3,1,2]);
condList = cat(3, condListP1, condListP2);
condList = permute(condList, [3,1,2]);

% Cut out all RTs that are < .5
y(abs(y)<.5) = nan;
minRT = .5

data = struct(...
    'nSubjects'       ,   size(subjList,2)      , ...
    'nAllTrials'      ,   160          , ...
    'y'               ,   y               , ...
    'condList'         ,   condList          , ...
    'groupList'       ,   group_list );



generator = @()struct(...
    'z'     , ceil(rand(2, size(subjList,2), 160, 1)) , ...
    'phi'   , rand(2, size(subjList,2), 1) , ...
    'alphagroupmid', (randi([10, 100], 1, 1))./100, ... % was: (randi([10, 100], nGroups, 1))./100, ...
    'taugroupmid', (randi([100, round(minRT*50000)], 1, 1))./100000, ... % was: (randi([1, round(minRT*50000)], nGroups, 1))./100000, ...
    'taugroupdiff', (randi([100, round(minRT*50000)], 1, 1))./300000, ...
    'betagroupmid', rand(1, 1)); ... % was:  rand(nGroups, 1), ...

disp('Running!')
tic
  [stats, chains, diagnostics, info] = callbayes('jags', ...
    'model'          ,     model , ...
    'data'           ,      data , ...
    'outputname'     , 'samples' , ...
    'init'           , generator , ...
    'nchains'        ,  4  , ...
    'nburnin'        ,  1  , ...
    'nsamples'       , 100  , ...
    'monitorparams'  ,    monitorParameters , ...
    'thin'           ,    1  , ...
    'workingdir'     ,   ['/tmp/' proj_id] , ...
    'verbosity'      ,        0  , ...
    'saveoutput'     ,     true  , ...
    'parallel'       ,1 , ...
    'modules'        ,  {'dic', 'wiener'}  );

% Print and save
fprintf('%s took %f seconds!\n', upper(engine), toc)
disp('Saving...')
save(strcat('../storage/',modelName,'_',subgroups(1),'_', subgroups(2), '.mat'), 'stats', 'chains', 'diagnostics', 'info', 'data', '-v7.3');
disp('Saved!')


%% Inspect the results
% First, inspect the convergence of each parameter
disp('Convergence statistics:')
grtable(chains, 1.05)
return;
%%
