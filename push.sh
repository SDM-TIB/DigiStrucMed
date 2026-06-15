#!/bin/bash

# Go to your project directory
cd C:\Users\tehranim.TIB.001\Desktop\Computer-20260504T112359Z-3-001\Computer\Master Thesis\EER-Questionaries\Questionnarie-KG

# Initialize git if not already initialized
git init

# Configure remote
git remote remove origin 2>/dev/null
git remote add origin https://github.com/SDM-TIB/DigiStrucMed.git

# Check current branch
git branch -M Develop-Questionnarie

# Add all files
git add .

# Commit
git commit -m "Update questionnaire KG pipeline"

# Push
git push -u origin Develop-Questionnarie