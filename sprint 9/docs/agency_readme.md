# Experiment Branching Workflow

1. **Start a New Experiment**  
   1.1 Switch to main: `git switch main`.  
   1.2 Pull the latest changes from remote: `git pull origin main`.  
   1.3 Create a new branch from main: `git switch -c experiments/EXP-XXX-Short-Description`.  

2. **Conduct Experiment**  
   2.1 Modify code to accomplish the test goal. ALL MODIFIED CODE MUST BE JUSTIFIED WITH COMMENTS. Ensure that results are automatically logged to `EXPERIMENTS.md` and `RESULTS.md`.  
   2.2 Update `Branch_Edit_Summary.md` with very detailed code changes made in this branch.  
   2.3 Run training (`python main.py`).

3. **Conclude Experiment**

   3.1 **If the experiment was successful**  
   3.1.1 Switch to experiment branch: `git switch experiments/EXP-XXX-...`  
   3.1.2 Add new files to git: `git add .`  
   3.1.3 Commit the changes. Note the ✅ in the commit message to indicate success:  
   `git commit -m "EXP-XXX: ✅ Description of results"`  
   3.1.4 Read `Branch_Edit_Summary.md`, `RESULTS.md`, and `experiment_log.md` to understand the changes made in this branch. 
   3.1.5 Switch to main: `git switch main`  
   3.1.6 Update `experiment_log.md` with EXPERIMENT_ID, all `Branch_Edit_Summary.md` data, all `RESULTS.md` data: `code experiment_log.md`  
   3.1.7 Stage the changes: `git add experiment_log.md`  
   3.1.8 Commit the changes. Note the ✅ in the commit message to indicate success:  
   `git commit -m "EXP-XXX: ✅ Description of results"`  
   3.1.9 Merge experiment: `git merge experiments/EXP-XXX-...`  
   3.1.10 Push: `git push origin main`

   3.2 **If Unsuccessful**  
   3.2.1 Switch to experiment branch: `git switch experiments/EXP-XXX-...`  
   3.2.2 Add new files to git: `git add .`  
   3.2.3 Commit and leave the branch as history. Note the ❌ in the commit message to indicate failure:  
   `git commit -m "EXP-XXX: ❌ Description of results"`  
   3.2.4 Read `Branch_Edit_Summary.md` to understand the changes made in this branch.  
   3.2.5 Switch to main: `git switch main`  
   3.2.6 Update `experiment_log.md` with EXPERIMENT_ID, all `Branch_Edit_Summary.md` data, all `RESULTS.md` data: `code experiment_log.md`  
   3.2.7 Stage the changes: `git add experiment_log.md`  
   3.2.8 Commit the changes. Note the ❌ in the commit message to indicate failure:  
   `git commit -m "EXP-XXX: ❌ Description of results"`  
   3.2.9 Go to step 1.

4. **Publishing**  
   4.1 “Publish Branch” in IDE pushes the experiment branch to remote (GitHub), ensuring it is backed up. Always say **Yes**.