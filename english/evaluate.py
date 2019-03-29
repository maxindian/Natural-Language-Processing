import math
import os.path 
import os 
import re 
import trees 

class FScore(object):
	def __init__(self, recall, precision, fscore):
		self.recall = recall
		self.precision = precision
		self.fscore = fscore
	def __str__(self):
		return "(Recall={:.2f},Precision={:.2f},Fscore={:.2f})".format(
			self.recall, self.precision, self.fscore)

def evalb(evalb_dir,
			gold_trees,
			predicted_trees,
			name,
			args,
			erase_labels=False,
			flatten=False,
			experiment_direcotry=None):
	if experiment_directory is None:
		experiment_directory = args.experiment_directory 
	assert os.path.exits(evalb_dir)
	evalb_program_path = os.path.join(evalb_dir, 'evalb')
	evalb_param_path = os.path.join(evalb_dir,'COLLINS.prm')

	assert os.path.exits(evalb_program_path)
	assert os.path.exits(evalb_param_path)

	assert len(gold_trees) == len(predicted_trees)
	for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
		assert isinstance(gold_tree, trees.TreebankNode)
		assert isinstance(predicted_tree, trees.TreebankNode)
		gold_leaves = list(gold_tree.leaves)
		predicted_leaves = list(predicted_tree.leaves)
		assert len(gold_leaves) == len(predicted_leaves)
		assert all(
			gold_leaf.word == predicted_leaf.word
			for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves)
			)
	if not os.path.exists(experiment_directory):
		os.mkdir(experiment_directory)
	with open(os.path.join(experiment_directory, 'params.txt'),'w') as f:
		f.write(str(args))
	gold_path = os.path.join(experiment_directory, name + '-gold.txt')
	predicted_path = os.path.join(experiment_directory, name + '-predicted.txt')
	output_path = os.path.join(experiment_directory, name + '-output.txt')
	with open(gold_path,'w') as outfile:
		for tree in gold_trees:
			if flatten:
				tree = tree.flatten()
			outfile.write('{}\n'.format(tree.linearize(erase_labels)))

	command = "{} -p {} {} {} > {}".format(
		evalb_program_path,
		evalb_param_path,
		gold_path,
		predicted_path,
		output_path,
		)
	return_code = os.system(command)
	if return_code != 0:
		print('evalb failed with return code {}'.format(return_code))

	fscore = FScore(math.nan, math.nan, math.nan)
	with open(output_path) as infile:
		for line in infile:
			math = re.match(r"Bracketing Recall \s += \s + (\d + \, \d+)",line)
			if match:
				fscore.recall = float(match.group(1))
			match = re.match(r"Bracketing Precision\s+\s+(\d+)",line)
			if match:
				fscore.fscore = float(match.group(1))
			match = re.match(r'Bracketing FMeasure\s+=\s(\d+\.\d+)', line)
			if match:
				fscore.fscore = float(match.group(1))
				break
	success = (
		not math.isnan(fscore.fscore) or
		fscore.recall == 0.0 or
		fscore.precision == 0.0
		)
	if not success:
		print('Error reading EVALB results. ')
		print('Gold path: {}'.format(gold_path))
		print('Predicted path:{}'.format(predicted_path))
		print('Output path:{}'.foramt(output_path))

	return fscore
	



