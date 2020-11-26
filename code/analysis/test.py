import h5py
import numpy as np
import matplotlib.pyplot as plt

f = h5py.File('../../storage/model_07_C1/samples_1_chain1.hdf5', 'r')
print(f.keys())
deltaDiffExercise = np.array(f.get("deltaDiffExercise"))
deltaDiffExerciseCarryover = np.array(f.get("deltaDiffExerciseCarryover"))
deltaDiffControl = np.array(f.get("deltaDiffControl"))
deltaMu = np.array(f.get("deltaMu"))
print(deltaMu.shape)

for param in ['alphaDiff', 'tauDiff', 'betaDiff']:
	plt.figure()
	plt.title("{}".format(param))
	plt.hist(np.array(f.get("{}Exercise".format(param))), bins=50)
	plt.savefig("{}diffExercise.png".format(param))
	plt.close()
	plt.figure()
	plt.title("{}".format(param))
	plt.hist(np.array(f.get("{}Control".format(param))), bins=50)
	plt.savefig("{}diffControl.png".format(param))
	plt.close()

f.close()

labels = ['Targ', 'HSim', 'LSim', 'Foil']


for i in range(4):
	plt.figure()
	plt.title("{} Exercise Diff".format(labels[i]))
	plt.hist(deltaDiffExercise[i,:], bins=50)
	plt.savefig("exercise_diff_{}.png".format(labels[i]))

for i in range(4):
	plt.figure()
	plt.title("{} Control Diff".format(labels[i]))
	plt.hist(deltaDiffControl[i,:], bins=50)
	plt.savefig("control_diff_{}.png".format(labels[i]))

'''
for i in range(4):
	plt.figure()
	plt.title("{} Carryover Diff".format(labels[i]))
	plt.hist(deltaDiffExerciseCarryover[i,:])
	plt.savefig("carryover_diff_{}.png".format(labels[i]))
'''


for group_idx, group_label in enumerate(['ExerciseGroup', 'ControlGroup']):
	for timepoint_idx in range(2):
		plt.figure()
		for cond_idx in range(4):
			plt.hist(deltaMu[group_idx,timepoint_idx,cond_idx,:], bins=100, alpha=.8)
		plt.title("{} P{}".format(group_label, timepoint_idx+1))
		plt.legend(labels)
		plt.savefig("deltagroup_{}_P{}.png".format(group_label,timepoint_idx+1))






