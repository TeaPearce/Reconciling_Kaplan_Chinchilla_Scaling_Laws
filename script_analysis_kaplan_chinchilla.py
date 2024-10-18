# Code for analysis and plots in the paper:
# "Reconciling Kaplan and Chinchilla Scaling Laws"
# Tim Pearce & Jinyeop Song

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 22})
plt.rcParams["font.family"] = "Times New Roman"

save_path = '/home/t-timpearce/wm_eval_toolkit/world-model-eval-toolkit/chin_kap_plots_clean/'

vocab_size = 32e3
aspect_ratio = 39.2 # d/L = 40, 39.2
# gamma = (aspect_ratio/12)**(1/3) * vocab_size # N_T = N_\e + gamma N_\e^{1/3}
gamma = 47491
aspect_ratio = 12*(gamma / vocab_size)**3
ctx=2048

coeffs = 'epoch' # 'epoch' or 'chin'
if coeffs == 'epoch':
    Nc=482.01
    Dc=2085.43
    alpha=0.3478
    beta=0.3658 
    E=1.8172
    print('using Epoch specification')
elif coeffs == 'chin':
    Nc=406.4
    Dc=410.7
    alpha=0.3392
    beta=0.2849 
    E=1.6934
    print('using Chinchilla specification')

def kaplan_chin_loss(N_T, D, Nc, Dc, alpha, beta, E):
	Loss = Nc / N_T**alpha + Dc / D**beta + E
	return Loss

print('target synth N propto C^a, a=', beta/(beta+alpha))
print('L propto C^b, b=', alpha*beta/(beta+alpha))

num_models = 20 # default 20
N_nonemb_all = np.logspace(2.9,9.2,num_models) # Kaplan paper "768 to 1.5 billion non-embedding parameters"
d_all = (N_nonemb_all/(12/aspect_ratio))**(1/3) # start from fixed aspect ratio
N_emb = vocab_size*d_all
N_T_all = N_nonemb_all + N_emb
D_all = np.logspace(6,25,1000)
N_T, D = np.meshgrid(N_T_all, D_all)
Loss = kaplan_chin_loss(N_T, D, Nc, Dc, alpha, beta, E)
colors_all = plt.cm.plasma(np.linspace(0,1,len(N_T_all)+3))

# plot chinchilla data and fit
parameters = [44, 57, 74, 90, 106, 117, 140, 163, 175, 196, 217, 251, 278, 306, 425, 489, 509, 552, 587, 632, 664, 724, 816, 893, 1018, 1143, 1266, 1424, 1429, 1593, 1609, 1731, 1794, 2007, 2283, 2298, 2639, 2980, 3530, 3802, 4084, 4516, 6796, 9293, 11452, 12295, 12569, 13735, 14494, 16183]
d_model = [512, 576, 640, 640, 640, 768, 768, 768, 896, 896, 896, 1024, 1024, 1024, 1280, 1280, 1408, 1280, 1408, 1536, 1408, 1536, 1536, 1792, 1792, 1792, 2048, 2176, 2048, 2048, 2176, 2304, 2176, 2304, 2304, 2560, 2560, 2560, 2688, 2816, 2944, 3072, 3584, 4096, 4352, 4608, 4608, 4864, 4992, 5120]
chin_data_orig = np.array([parameters, d_model]).T
chin_data_non = chin_data_orig.copy()
chin_data_non[:,0] = chin_data_non[:,0]*1e6
# chin_data_non[:,1] = chin_data_non[:,1]*(32000+2048) # report 32k and position encodings
chin_data_non[:,1] = chin_data_non[:,1]*(vocab_size) # report 32k
chin_data_non[:,1] = chin_data_non[:,0]-chin_data_non[:,1] # could add this line to get non-embed params 

fig_dims = (15,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
ax.set_xlabel('$N_T$, total parameters')
ax.set_ylabel('$N_{\setminus E}$, non-embedding parameters')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(which='major', linestyle='-', linewidth='0.8', color='black')
ax.grid(which='minor', linestyle='-', linewidth='0.3', color='black')
x_fit = np.logspace(5,11,1000)
ax.scatter(chin_data_non[:,0],chin_data_non[:,1],label='Chinchilla model configs', color='blue', s=50, zorder=10)
ax.plot(x_fit,x_fit,label='$N_T = N_{\setminus E}$', color='gray', linewidth=3, linestyle='--', zorder=8)
y_fit_2 = x_fit + gamma*x_fit**(1/3)
ax.plot(y_fit_2, x_fit,label='Fit, $N_T = N_{\setminus E} + 47491 N_{\setminus E}^{1/3}$', color='fuchsia', linewidth=3, linestyle='--',zorder=9)
# y_fit_2 = x_fit + 47491*x_fit**(0.34)
# ax.plot(y_fit_2, x_fit,label='$N_T = N_{ | E} + 47491 N_{|E}^{0.34}$', color='purple', linewidth=3, linestyle='--',zorder=9)
# y_fit_2 = x_fit + 52960*x_fit**(1/3)
# ax.plot(y_fit_2, x_fit,label='$N_T = N_{ | E} + 52960 N_{|E}^{1/3}$', color='gold', linewidth=3, linestyle='--',zorder=9)
ax.legend(loc='upper left', prop={'size': 20})
ax.set_xlim([1e6,2e10])
ax.set_ylim([1e5,2e10])
fig.tight_layout()
fig.savefig(os.path.join(save_path,'chin_data_NE_vs_NT_fit.png'))
plt.close()

# plot Loss vs D for each N_nonE
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_nonemb_all)):
	if N_nonemb_all[i]/1e6>0.1:
		ax.plot(D_all, Loss[:,i], label=f'$N_{{\setminus E}} = {round(N_nonemb_all[i]/1e6,2)}$M', color=colors_all[i])
	else:
		ax.plot(D_all, Loss[:,i], label=f'$N_{{\setminus E}} = {round(N_nonemb_all[i]/1e3,2)}$K', color=colors_all[i])
ax.set_xlabel('$D$')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([D_all[0],1e18])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(fontsize='x-small', loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_D_vs_NE.png'))
plt.close()

# plot Loss vs D for fixed N_T
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_T_all)):
	if N_T_all[i]/1e6>0.1:
		ax.plot(D_all, Loss[:,i], label=f'$N_T = {round(N_T_all[i]/1e6,2)}$M', color=colors_all[i])
	else:
		ax.plot(D_all, Loss[:,i], label=f'$N_T = {round(N_T_all[i]/1e3,2)}$K', color=colors_all[i])
ax.set_xlabel('D')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([D_all[0],1e18])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(fontsize='x-small', loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_D_vs_NT.png'))
plt.close()

# calculate compute and optimal params
info_arr = []
for i in range(len(N_T_all)):
	arr_zeros = Loss[:,i]*0.
	C_T_accurate = 6*N_T_all[i]*D_all + 12*ctx/aspect_ratio*d_all[i]**2*D_all
	C_T_approx = 6*N_T_all[i]*D_all
	C_nonE_accurate = 6*N_nonemb_all[i]*D_all + 12*ctx/aspect_ratio*d_all[i]**2*D_all
	C_nonE_approx = 6*N_nonemb_all[i]*D_all
	info_arr.append([arr_zeros+N_T_all[i], arr_zeros+N_nonemb_all[i], arr_zeros+N_emb[i], D_all, arr_zeros+d_all[i], C_T_accurate, C_T_approx, C_nonE_accurate, C_nonE_approx, Loss[:,i]])
    # 0 N_T, 1 N_\E, 2 N_E, 3 D, 4 d, 5 C_T_accurate, 6 C_T_approx, 7 C_\E_accurate, 8 C_\E_approx, 9 Loss
	
# calculate C_T and N^*_T 
info_arr = np.array(info_arr)
# c_range_bounds = [13.4, 20.7]
c_range_bounds = [14, 20.7]
C_range_T = np.logspace(c_range_bounds[0],c_range_bounds[1],100)
best_loss = []
for C_i in C_range_T:
	poss_loss1 = []
	for i in range(len(N_T_all)):
		# change index depending on which C to use
		N_i = N_T_all[i] # N_T
		loss_diff = np.abs(info_arr[i,6] - C_i) # C_\C_T_approx
		loss_diff_min = np.min(loss_diff)
		loss_diff_min_idx = np.argmin(loss_diff)
		poss_loss1.append([info_arr[i,9][loss_diff_min_idx], N_i])
	poss_loss1 = np.array(poss_loss1)
	# find lowest loss and model
	best_idx = np.argmin(poss_loss1[:,0])
	best_loss.append(poss_loss1[best_idx])
best_loss_T = np.array(best_loss)

# calculate C_nonE and N^*_nonE
info_arr = np.array(info_arr)
c_range_bounds = [12.95, 20.7]
C_range = np.logspace(c_range_bounds[0],c_range_bounds[1],100)
best_loss = []
for C_i in C_range:
	poss_loss1 = []
	for i in range(len(N_T_all)):
		# change index depending on which C to use
		N_i = N_nonemb_all[i] # N_\E
		loss_diff = np.abs(info_arr[i,8] - C_i) # C_\E_approx
		loss_diff_min = np.min(loss_diff)
		loss_diff_min_idx = np.argmin(loss_diff)
		poss_loss1.append([info_arr[i,9][loss_diff_min_idx], N_i])
	poss_loss1 = np.array(poss_loss1)
	# find lowest loss and model
	best_idx = np.argmin(poss_loss1[:,0])
	best_loss.append(poss_loss1[best_idx])
best_loss_nonE = np.array(best_loss)

# plot N_nonE vs C_nonE
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_nonemb_all)):
	label_i = None
	ax.plot(6*N_nonemb_all[i]*D_all, Loss[:,i], label=label_i, color=colors_all[i])
ax.scatter(C_range, best_loss_nonE[:,0], label='Compute efficient frontier', color='red',s=10, zorder=10)
ax.set_xlabel('$C_{\setminus E}$')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e10,1e23])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_NE_vs_CE.png'))
plt.close()

# plot N_nonE vs C_nonE with fit
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_nonemb_all)):
	label_i = None
	ax.plot(6*N_nonemb_all[i]*D_all, Loss[:,i], label=label_i, color=colors_all[i])
ax.scatter(C_range, best_loss_nonE[:,0], label='Compute efficient frontier', color='red',s=10, zorder=10)
# fit L = bC^m
m, b = np.polyfit(np.log(C_range[:]), np.log(best_loss_nonE[:,0]), 1)
print('C_nonE, Kaplan compute-loss form',m)
x_fit = np.logspace(9,25,100)
y_fit = np.exp(m*np.log(x_fit) + b)
ax.plot(x_fit,y_fit,label=f'Kaplan compute-loss form:\n $log (L^*_{{\setminus E}}) = {m:.3f} log(C_{{\setminus E}}) + {b:.2f}$', color='blue', linestyle='--', linewidth=3,zorder=8)

# fit L - E = bC^m
m, b = np.polyfit(np.log(C_range[:]), np.log(best_loss_nonE[:,0]-E), 1)
print('C_nonE, Chinchilla compute-loss form, L-E',m)
x_fit = np.logspace(9,25,100)
y_fit = np.exp(m*np.log(x_fit) + b) + E
ax.plot(x_fit,y_fit,label=f'Chinchilla compute-loss form:\n $log (L^*_{{\setminus E}}) - {E:.2f} = {m:.3f} log(C_{{\setminus E}}) + {b:.2f}$', color='green', linestyle='--', linewidth=3,zorder=8)

ax.set_xlabel('$C_{\setminus E}$')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e10,1e23])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_NE_vs_CE_fit.png'))
plt.close()


# plot of coefficient, N^*_nonE and C_nonE
fig_dims = (15,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
ax.scatter(C_range[:], best_loss_nonE[:,1], label='Compute efficient frontier', color='red',s=30,zorder=12)
N_nonemb_all_1 = np.logspace(2,11,100)
C_pred = N_nonemb_all_1*(N_nonemb_all_1 + (gamma/3)*N_nonemb_all_1**(1/3))**(-1/beta) * (N_nonemb_all_1 + gamma*N_nonemb_all_1**(1/3))**((1+alpha)/beta) * (beta/alpha * Dc/Nc)**(1/beta)*6
ax.plot(C_pred[:], N_nonemb_all_1[:], label='Analytical $N^*_{{\setminus E}}$ vs. $C_{{\setminus E}}$', color='black', linewidth=3, linestyle='--',zorder=10)
# ax.scatter(C_pred[17:81], N_nonemb_all_1[17:81], label='Analytical $N^*_{{\setminus E}}$ vs. $C_{{\setminus E}}$ used for fit', color='blue', linewidth=3, linestyle='--',zorder=10)
m, b = np.polyfit(np.log(C_range[:]), np.log(best_loss_nonE[:,1]), 1)
print('main coeff: slope on loss vs N_opt, simulated points',m)
# m_analy, b_analy = np.polyfit(np.log(C_pred[:]), np.log(N_nonemb_all_1[:]), 1) # fit on analytical, gives slight difference as logarithmically spaced on y not x axis
# print('main coeff: slope on loss vs N_opt, analytical points',m_analy)
x_fit = np.logspace(11,25,100)
y_fit = np.exp(m*np.log(x_fit) + b)
ax.plot(x_fit,y_fit,label=f'Local fit to frontier: $log (N^*_{{\setminus E}}) = {m:.2f} log(C_{{\setminus E}}) {b:.2f}$', color='blue', linestyle='-', linewidth=3,zorder=11)
m, b = np.polyfit(np.log(C_pred[90:]), np.log(N_nonemb_all_1[90:]), 1)
print('main coeff: slope on loss vs N_opt',m)
y_fit = np.exp(m*np.log(x_fit) + b)
ax.plot(x_fit,y_fit,label=f'Large $N_{{\setminus E}}$ regime fit: $log (N^*_{{\setminus E}}) = {m:.2f} log(C_{{\setminus E}}) {b:.2f}$', color='orange', linestyle='-', linewidth=3,zorder=8)
ax.set_xlabel('$C_{\setminus E}$')
ax.set_ylabel('$N^*_{\setminus E}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e11,1e24])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_synth_NE_CE_coeff_'+coeffs+'.png'))
plt.close()


# plot N_T vs C_T
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_T_all)):
	label_i = None
	ax.plot(6*N_T_all[i]*D_all, Loss[:,i], label=label_i, color=colors_all[i])
ax.scatter(C_range_T, best_loss_T[:,0], label='Compute efficient frontier', color='red',s=10, zorder=10)
ax.set_xlabel('$C_T$')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e13,1e23])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
# ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_NT_vs_CT.png'))
plt.close()

# plot N_T vs C_T  with fit
fig_dims = (10,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
for i in range(len(N_T_all)):
	label_i = None
	ax.plot(6*N_T_all[i]*D_all, Loss[:,i], label=label_i, color=colors_all[i])
ax.scatter(C_range_T, best_loss_T[:,0], label='Compute efficient frontier', color='red',s=10, zorder=10)
# fit L = bC^m
m, b = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:,0]), 1)
print('C_T, Kaplan compute-loss form',m)
x_fit = np.logspace(13,25,100)
y_fit = np.exp(m*np.log(x_fit) + b)
ax.plot(x_fit,y_fit,label=f'Kaplan compute-loss form:\n $log (L^*_T) = {m:.3f} log(C_T) + {b:.2f}$', color='blue', linestyle='--', linewidth=3,zorder=8)

# fit L - E = bC^m
m, b = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:,0]-E), 1)
print('C_T, Chinchilla compute-loss form, L-E',m)
x_fit = np.logspace(13,25,100)
y_fit = np.exp(m*np.log(x_fit) + b) + E
ax.plot(x_fit,y_fit,label=f'Chinchilla compute-loss form:\n $log (L^*_T) - {E:.2f} = {m:.3f} log(C_T) + {b:.2f}$', color='green', linestyle='--', linewidth=3,zorder=8)

ax.set_xlabel('$C_T$')
ax.set_ylabel('Loss')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e13,1e23])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
# ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_loss_NT_vs_CT_fit.png'))
plt.close()


# plot of coefficient, N^*_T and C_T
fig_dims = (15,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
ax.scatter(C_range_T[:], best_loss_T[:,1], label='Compute efficient frontier', color='red',s=30,zorder=12)
# should prob add analytical line here too ...
m, b = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:,1]), 1)
print('main coeff: slope on loss vs N_opt, total params',m)
x_fit = np.logspace(11,25,100)
y_fit = np.exp(m*np.log(x_fit) + b)
ax.plot(x_fit,y_fit,label=f'Local fit to frontier: $log (N^*_{{T}}) = {m:.2f} log(C_{{T}}) {b:.2f}$', color='deeppink', linestyle='-', linewidth=3,zorder=11)
ax.set_xlabel('$C_{T}$')
ax.set_ylabel('$N^*_{T}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e11,1e24])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
# ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_synth_NT_CT_coeff_'+coeffs+'.png'))
plt.close()


# plot eq and gradient g
fig_dims = (15,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
N_nonemb_all_1 = np.logspace(2,11,100)
C_pred = N_nonemb_all_1*(N_nonemb_all_1 + (gamma/3)*N_nonemb_all_1**(1/3))**(-1/beta) * (N_nonemb_all_1 + gamma*N_nonemb_all_1**(1/3))**((1+alpha)/beta) * (beta/alpha * Dc/Nc)**(1/beta)*6
ax.plot(C_pred[:], N_nonemb_all_1[:], label=r'$N_{\setminus E}$', color='black', linewidth=3, linestyle='--',zorder=11)
# g
def fn_g_recip(x, gamma, beta, alpha):
    term1 = 1
    term2 = - (1 / beta) * ((x ** (2/3) + (gamma / 9)) / (x ** (2/3) + (gamma / 3)))
    term3 = ((alpha + 1) / beta) * ((x ** (2/3) + (gamma / 3)) / (x ** (2/3) + gamma))
    result = term1 + term2 + term3
    return result

g_all = fn_g_recip(N_nonemb_all_1, gamma, beta, alpha)
# plot on second axis
ax2 = ax.twinx()
ax2.plot(C_pred, 1/g_all, label='$g$', color='dodgerblue', linewidth=3, linestyle='--',zorder=11)
# limits
ax2.axhline(y=beta/(beta+alpha), color='grey', linestyle='--', linewidth=2, zorder=8, label=r'$\beta/(\beta + \alpha)$')
ax2.axhline(y=beta/(alpha/3+beta), color='grey', linestyle=':', linewidth=2, zorder=8, label=r'$\beta/(\alpha/3+\beta)$')
ax2.set_ylabel('$g$')
ax2.get_yaxis().get_major_formatter().set_scientific(False)
# combine legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
ax.set_xlabel(r'$C_{ \setminus E}$')
ax.set_ylabel(r'$N^*_{ \setminus E}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([C_pred[0],C_pred[-1]])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
# ax.grid(which='minor', linestyle='--', linewidth='0.2', color='black')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_eqs_only.png'))
plt.close()


# derivative of d log L_\E* / d log C_\E

def fn_Nopt_to_C(x, gamma, alpha, beta, Nc, Dc):
	# use eq 16 to convert from N^*_\E to C_\E
	# x is N_\E
	term1 = 6 * x * (x + gamma/3*x**(1/3))**(-1/beta)
	term2 = (x + gamma*x**(1/3))**((1+alpha)/beta)
	term3 = ((beta/alpha) * (Dc/Nc))**(1/beta)
	C = term1 * term2 * term3
	return C

def fn_k_inner(x, y, gamma, alpha, beta, Nc, Dc):
	# x is N_\E
	# y is C_\E
    term1 = -alpha * (Nc * (x + gamma/3*x**(1/3))) / ((x + gamma*x**(1/3))**(alpha+1))
    term2 = beta * Dc * (y / (6*x))**(-beta)
    result = term1 + term2
    return result

def fn_L_opt(x, y, gamma, alpha, beta, Nc, Dc, E):
	# x is N_\E
	# y is C_\E
	term1 = Nc / (x + gamma*x**(1/3))**alpha
	term2 = Dc / (y / (6*x))**beta
	result = term1 + term2 + E
	return result

def fn_extra(x, y, gamma, alpha, beta, Nc, Dc, E):
	# x is N_\E
	# y is C_\E
	term2 = beta * Dc * (y / (6*x))**(-beta)
	return term2

C_all = []
k_all = []
for N in N_nonemb_all_1:
	C = fn_Nopt_to_C(N, gamma, alpha, beta, Nc, Dc)
	part1 = fn_k_inner(N, C, gamma, alpha, beta, Nc, Dc)
	g = 1/fn_g_recip(N, gamma, beta, alpha)
	L_opt = fn_L_opt(N, C, gamma, alpha, beta, Nc, Dc, E)
	extra_term = fn_extra(N, C, gamma, alpha, beta, Nc, Dc, E)
	k = part1*g/L_opt - extra_term/L_opt
	C_all.append(C)
	k_all.append(k)

fig_dims = (15,10)
fig, ax = plt.subplots(1,1,figsize=fig_dims)
# ax.plot(C_all, k_all, label=r'Analytical $ d \log L_{\setminus E}^* / d \log C_{\setminus E}$', color='dodgerblue', linewidth=3, linestyle='-',zorder=11)
ax.plot(C_all, k_all, label=r'Analytical $k$', color='dodgerblue', linewidth=3, linestyle='-',zorder=11)
ax.set_xlabel(r'$C_{\setminus E}$')
# ax.set_ylabel(r'$ d \log L_{\setminus E}^* / d \log C_{\setminus E}$')
ax.set_ylabel(r'$k$')
# add horizontal lines
ax.axhline(y=-0.057, color='grey', linestyle='--', linewidth=2, zorder=8, label=r'Kaplan reported slope, $\gamma=-0.057$')
ax.axhline(y=-0.069, color='green', linestyle='--', linewidth=2, zorder=8, label=r'Chinchilla simulated fitted slope, $\gamma=-0.069$')
# ax.axhline(y=-0.057, color='grey', linestyle='--', linewidth=2, zorder=8, label=r'Kaplan reported slope, $\gamma=-0.057$')
ax.legend()
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim([C_all[0],C_all[-1]])
# ax.set_ylim([-0.1,0.])
ax.grid(which='major', linestyle='--', linewidth='0.5', color='black')
fig.tight_layout()
fig.savefig(os.path.join(save_path,'kaplan_chin_dlogLdlogC.png'))
plt.close()





