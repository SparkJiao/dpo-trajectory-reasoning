import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter



scale 	= [str(100 * (i + 1)) for i in range(5)]
colors 	= ['#9E940E', '#64B69B', '#866897']
colors	= ['#877EF8', '#FF8488', '#9E940E']
# colors  = ['#CF0064', '#E68B0B']

# roberta 	= [43.26, 45.36, 51.84, 52.96, 55.02]
# lreasoner	= [45.92, 52.98, 56.08, 58.02, 58.30]
# merit		= [47.88, 55.24, 57.06, 59.94, 61.62]

# roberta_h	= [30.82, 31.89, 36.89, 35.89, 37.68]
# lreasoner_h	= [38.00, 42.46, 44.02, 44.32, 43.11]
# merit_h		= [40.03, 44.75, 47.00, 46.82, 47.75]

# roberta_e 	= [59.09, 62.50, 70.86, 74.68, 77.09]
# lreasoner_e = [56.00, 66.36, 71.45, 75.45, 77.63]
# merit_e		= [57.86, 68.59, 69.86, 76.63, 79.27]

# merit_dev 	= [68.20, 68.32, 68.32, 67.68, 69.36]
# merit_test 	= [61.70, 61.50, 61.80, 61.84, 61.62]


# tf_ab_dev = [77.06, 76.49, 76.52, 76.90, 77.06]
# tf_ab_test = [76.39, 76.64, 76.36, 76.88, 76.61]

# gnn_ab_dev = [77.06, 76.77, 77.07, 77.01, 77.42]
# gnn_ab_test = [76.39, 76.43, 76.43, 76.77, 76.82]

tf_ab_dev = [76.85, 77.07, 77.1, 76.71, 76.44]
tf_ab_test = [77.30, 76.45, 76.63, 77.05, 77.10]

gnn_ab_dev = [76.85, 77.03, 77.27, 77.08, 77.18]
gnn_ab_test = [77.30, 77.24, 76.55, 76.32, 76.96]

#---------------------------------- Line Chart -----------------------------#

font = {
		# 'font.family': 'Serif',
		# 'font.weight': 'bold',
		'font.size': 10
		}
plt.rcParams.update(font)

fig = plt.figure(figsize=(12, 3), dpi=200)
axe0 = fig.add_subplot(131)
axe1 = fig.add_subplot(133)
# axe2 = fig.add_subplot(133)


def plot_subplot(ax, title, lines, scale, labels, set_y_label: bool = True, set_x_label: bool = True):
	""" line list must in the same order. """
	# ax.set_title(title)
	if set_y_label:
		ax.set_ylabel('AUC (%)')
	ax.set_xlim([0, len(lines[0]) - 1])
	ax.set_ylim([75, 78])
	if set_x_label:
		ax.set_xlabel('Ratio of Training Data')

	for axis in ['top','bottom','left','right']:
  		ax.spines[axis].set_linewidth(1.2)

	candidate_markers = ['o', 'v', 's', 'D', 'p', '*', '+']
	for idx, line in enumerate(lines):
		# line = gaussian_filter(line, 1)
		ax.plot(scale, line,
			label=labels[idx],
			linewidth=1.8,
			markersize=7,
			marker=candidate_markers[idx],
			color=colors[idx],
		)

	ax.grid(axis='y', linestyle='--')

	# ax.legend(loc='best', bbox_to_anchor=anchor, fontsize=9)
	ax.legend(loc='lower right', fontsize=10)


# plot_subplot(
# 	ax 		= axe0,
# 	title	= 'Test',
# 	lines 	= [roberta, lreasoner, merit],
# 	labels 	= ['RoBERTa', 'LReasoner', 'MERIt + Prompt'],
# 	)
# plot_subplot(
# 	ax 		= axe1,
# 	title	= 'Test-H',
# 	lines 	= [roberta_h, lreasoner_h, merit_h],
# 	labels  = ['RoBERTa', 'LReasoner', 'MERIt + Prompt'],
# 	set_y_label = False
# 	)
# plot_subplot(
# 	ax 		= axe2,
# 	title	= 'Test-E',
# 	lines 	= [roberta_e, lreasoner_e, merit_e],
# 	labels 	= ['RoBERTa', 'LReasoner', 'MERIt + Prompt'],
# 	set_y_label = False,
# 	set_x_label = False
# 	)

plot_subplot(
	ax 		= axe0,
	title	= 'Transformer Layers',
	lines 	= [tf_ab_dev, tf_ab_test],
	scale   = list(map(str, [1, 2, 3, 4, 5])),
	labels 	= ['Val', 'Test'],
	set_x_label = False
	)
# colors	= ['#FF8488', '#9E940E']
plot_subplot(
	ax 		= axe1,
	title	= 'GNN Layers',
	lines 	= [gnn_ab_dev, gnn_ab_test],
	scale   = list(map(str, [2, 3, 4, 5, 6])),
	labels  = ['Val', 'Test'],
	# set_y_label = False,
	set_x_label = False
	)

plt.show()
# plt.subplots_adjust(
# 	left=0.08,
# 	bottom=0.07,
# 	right=0.97,
# 	top=0.88,
# 	wspace=0.20,
# 	hspace=0.51
# )
# plt.savefig('../writing/images/line.pdf')

