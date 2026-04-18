import os
import glob
import pandas as pd
import numpy as np
import ast

def main():
    # ==========================================
    # 1. Path configuration
    # ==========================================
    rl_ckpts_dir = './results/ddpo_train_600'
    csv_out_dir = os.path.join(rl_ckpts_dir, 'val_csv_results')
    
    detail_files = glob.glob(os.path.join(csv_out_dir, 'val_detail_epoch_*.csv'))
    if not detail_files:
        print(f"❌ Detail files not found, please check path: {csv_out_dir}")
        return

    def sort_key(filepath):
        epoch_str = os.path.basename(filepath).replace('val_detail_epoch_', '').replace('.csv', '')
        return -1 if epoch_str == 'Baseline' else int(epoch_str)

    detail_files = sorted(detail_files, key=sort_key)

    # ==========================================
    # 2. Data loading and 6qzh cleanup
    # ==========================================
    print("🧹 [Stage 1] Loading all data and removing 6qzh...")
    all_samples = []  # Store all samples: (epoch_str, rmsds, vinas, confs)
    
    for f in detail_files:
        epoch_str = os.path.basename(f).replace('val_detail_epoch_', '').replace('.csv', '')
        df = pd.read_csv(f)
        
        # Remove 6qzh entries
        if '6qzh' in df['complex_name'].values:
            df = df[df['complex_name'] != '6qzh']
            df.to_csv(f, index=False)
            
        for idx, row in df.iterrows():
            if 'all_32_rmsds' not in row or pd.isna(row['all_32_rmsds']):
                continue
            try:
                rmsds = np.array(ast.literal_eval(str(row['all_32_rmsds'])))
                vinas = np.array(ast.literal_eval(str(row['all_32_vinas'])))
                confs = np.array(ast.literal_eval(str(row['all_32_confs'])))
                all_samples.append((epoch_str, rmsds, vinas, confs))
            except:
                pass

    total_samples = len(all_samples)
    print(f"✅ Successfully loaded {total_samples} validation samples.\n")

    # ==========================================
    # 3. Global grid search - optimize both top-1 and top-5
    # ==========================================
    print("🚀 [Stage 2] Searching global pool for optimal hyperparameters...")
    rmsd_threshold = 2.0
    
    # Search best w (weighted method)
    best_w = 0.0
    best_w_acc_top1 = -1.0
    best_w_acc_top5 = -1.0
    
    for w in np.linspace(0.0, 1.0, 41):  # step size 0.05
        successes_top1 = 0
        successes_top5 = 0
        
        for _, rmsds, vinas, confs in all_samples:
            hybrid_scores = w * vinas + (1.0 - w) * confs
            
            # top-1 success rate
            if rmsds[np.argmax(hybrid_scores)] < rmsd_threshold:
                successes_top1 += 1
            
            # top-5 success rate
            top5_indices = np.argsort(hybrid_scores)[::-1][:5]
            if np.any(rmsds[top5_indices] < rmsd_threshold):
                successes_top5 += 1
                
        acc_top1 = successes_top1 / total_samples * 100
        acc_top5 = successes_top5 / total_samples * 100
        
        if acc_top1 > best_w_acc_top1:
            best_w_acc_top1 = acc_top1
            best_w_top1 = w
            best_w_acc_top5_top1 = acc_top5  # top-5 success paired with best top-1
            
        if acc_top5 > best_w_acc_top5:
            best_w_acc_top5 = acc_top5
            best_w_top5 = w
            best_w_acc_top1_top5 = acc_top1  # top-1 success paired with best top-5

    # Search best z (funnel method) - optimize both top-1 and top-5
    best_z_top1 = 1
    best_z_acc_top1 = -1.0
    best_z_acc_top5_at_top1 = -1.0  # top-5 success paired with best top-1
    
    best_z_top5 = 1
    best_z_acc_top5 = -1.0
    best_z_acc_top1_at_top5 = -1.0  # top-1 success paired with best top-5
    
    for z in range(1, 33):  # search range expanded to 1-32
        successes_top1 = 0
        successes_top5 = 0
        
        for _, rmsds, vinas, confs in all_samples:
            top_z_conf_indices = np.argsort(confs)[::-1][:z]
            
            # top-1 success rate
            if len(top_z_conf_indices) > 0:
                best_relative_idx = np.argmax(vinas[top_z_conf_indices])
                best_absolute_idx = top_z_conf_indices[best_relative_idx]
                if rmsds[best_absolute_idx] < rmsd_threshold:
                    successes_top1 += 1
            
            # top-5 success rate
            if len(top_z_conf_indices) > 0:
                # In top-z by confidence, choose top-5 by vina score
                vina_scores_in_topz = vinas[top_z_conf_indices]
                top5_in_topz_indices = np.argsort(vina_scores_in_topz)[::-1][:5]
                top5_absolute_indices = top_z_conf_indices[top5_in_topz_indices]
                
                if np.any(rmsds[top5_absolute_indices] < rmsd_threshold):
                    successes_top5 += 1
        
        acc_top1 = successes_top1 / total_samples * 100
        acc_top5 = successes_top5 / total_samples * 100
        
        # Update best top-1 parameter
        if acc_top1 > best_z_acc_top1:
            best_z_acc_top1 = acc_top1
            best_z_top1 = z
            best_z_acc_top5_at_top1 = acc_top5
        
        # Update best top-5 parameter
        if acc_top5 > best_z_acc_top5:
            best_z_acc_top5 = acc_top5
            best_z_top5 = z
            best_z_acc_top1_at_top5 = acc_top1

    print("\n🏆 [Global Best Parameters - Weighted Method]")
    print(f"  TOP-1 best: w = {best_w_top1:.2f}*Vina + {1-best_w_top1:.2f}*Conf")
    print(f"    Success: TOP-1 = {best_w_acc_top1:.2f}%, TOP-5 = {best_w_acc_top5_top1:.2f}%")
    print(f"  TOP-5 best: w = {best_w_top5:.2f}*Vina + {1-best_w_top5:.2f}*Conf")
    print(f"    Success: TOP-1 = {best_w_acc_top1_top5:.2f}%, TOP-5 = {best_w_acc_top5:.2f}%")
    
    print("\n🏆 [Global Best Parameters - Funnel Method]")
    print(f"  TOP-1 best: take top {best_z_top1} by Conf -> pick highest Vina")
    print(f"    Success: TOP-1 = {best_z_acc_top1:.2f}%, TOP-5 = {best_z_acc_top5_at_top1:.2f}%")
    print(f"  TOP-5 best: take top {best_z_top5} by Conf -> pick top-5 highest Vina")
    print(f"    Success: TOP-1 = {best_z_acc_top1_at_top5:.2f}%, TOP-5 = {best_z_acc_top5:.2f}%")
    
    # Use top-1 optimal parameters for downstream analysis
    print(f"\n📊 Use TOP-1 optimal parameters for epoch trend analysis:")
    print(f"  Weighted method: w = {best_w_top1:.3f}")
    print(f"  Funnel method: z = {best_z_top1}")

    # ==========================================
    # 4. Reconstruct per-epoch performance with globally optimal top-1 parameters
    # ==========================================
    print(f"\n📈 [Stage 3] Reconstruct epoch trend with TOP-1 optimal parameters (TOP-1 success):")
    print(f"{'Epoch':<10} | {'Weighted(w={:.3f})':<20} | {'Funnel(z={})':<20}".format(
        best_w_top1, best_z_top1))
    print("-" * 55)
    
    # Group statistics by epoch
    epoch_dict = {}
    for ep, rmsds, vinas, confs in all_samples:
        if ep not in epoch_dict:
            epoch_dict[ep] = []
        epoch_dict[ep].append((rmsds, vinas, confs))
        
    for f in detail_files:
        ep = os.path.basename(f).replace('val_detail_epoch_', '').replace('.csv', '')
        if ep not in epoch_dict: continue
        
        ep_samples = epoch_dict[ep]
        n_ep = len(ep_samples)
        
        # Compute weighted-method success (top-1) for this epoch
        w_succ = 0
        for rmsds, vinas, confs in ep_samples:
            hybrid_scores = best_w_top1 * vinas + (1.0 - best_w_top1) * confs
            if rmsds[np.argmax(hybrid_scores)] < rmsd_threshold:
                w_succ += 1
                
        # Compute funnel-method success (top-1) for this epoch
        z_succ = 0
        for rmsds, vinas, confs in ep_samples:
            top_z_idx = np.argsort(confs)[::-1][:best_z_top1]
            if len(top_z_idx) > 0:
                best_in_topz = np.argmax(vinas[top_z_idx])
                if rmsds[top_z_idx[best_in_topz]] < rmsd_threshold:
                    z_succ += 1
                
        w_acc = w_succ / n_ep * 100
        z_acc = z_succ / n_ep * 100
        print(f"{ep:<10} | {w_acc:>18.2f}% | {z_acc:>18.2f}%")
    
    # Optional: also show TOP-5 success trend
    print(f"\n📈 Reconstruct epoch trend with TOP-1 optimal parameters (TOP-5 success):")
    print(f"{'Epoch':<10} | {'Weighted(w={:.3f})':<20} | {'Funnel(z={})':<20}".format(
        best_w_top1, best_z_top1))
    print("-" * 55)
    
    for f in detail_files:
        ep = os.path.basename(f).replace('val_detail_epoch_', '').replace('.csv', '')
        if ep not in epoch_dict: continue
        
        ep_samples = epoch_dict[ep]
        n_ep = len(ep_samples)
        
        # Compute weighted-method success (top-5) for this epoch
        w_succ = 0
        for rmsds, vinas, confs in ep_samples:
            hybrid_scores = best_w_top1 * vinas + (1.0 - best_w_top1) * confs
            top5_indices = np.argsort(hybrid_scores)[::-1][:5]
            if np.any(rmsds[top5_indices] < rmsd_threshold):
                w_succ += 1
                
        # Compute funnel-method success (top-5) for this epoch
        z_succ = 0
        for rmsds, vinas, confs in ep_samples:
            top_z_idx = np.argsort(confs)[::-1][:best_z_top1]
            if len(top_z_idx) > 0:
                vina_scores_in_topz = vinas[top_z_idx]
                top5_in_topz_indices = np.argsort(vina_scores_in_topz)[::-1][:5]
                top5_absolute_indices = top_z_idx[top5_in_topz_indices]
                if np.any(rmsds[top5_absolute_indices] < rmsd_threshold):
                    z_succ += 1
                
        w_acc = w_succ / n_ep * 100
        z_acc = z_succ / n_ep * 100
        print(f"{ep:<10} | {w_acc:>18.2f}% | {z_acc:>18.2f}%")

if __name__ == '__main__':
    main()