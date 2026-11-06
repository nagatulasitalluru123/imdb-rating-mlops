# Lab 3 — Git-Based Version Control and Collaborative ML Workflows

Suggested sequence:

```bash
git init
git add .
git commit -m "chore: initialize IMDb ML project"
git branch lab3-version-control
git switch lab3-version-control
# make a meaningful change
git add .
git commit -m "feat: add IMDb baseline training workflow"
git switch main
git merge lab3-version-control
# if a conflict occurs: edit files, then
git add .
git commit -m "fix: resolve merge conflict"
# rollback/revert example
git revert <commit-hash>
# tag a lab milestone
git tag -a lab3-v1.0 -m "IMDb Lab 3 complete"
git log --oneline --decorate --graph --all
```

Use meaningful commits and keep one conceptual change per commit where practical.
