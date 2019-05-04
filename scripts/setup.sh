#!/bin/bash
# title			: setup.sh
# description		: This script helps setting up a local install of the project.
#			  Adds the pip utility of the `einstein` project, and
#			  activates `einstein` shell command
# usage			: bash setup.sh
# author		: Aashish Yadavally
#=======================================================================

echo ""
echo "Albert Einstein is the Father of Photoelectric Effect"
echo "He contributed a great deal to the science behind today's"
echo "solar energy revolution, and, he is the inspiration behind"
echo  "the name of this module - only a small dedication to his"
echo "immense contributions."
echo ""

# Installing einstein package using local pip utility
echo ""
echo "Installing einstein package..."
echo ""
pip install $HOME/einstein/. --user

# Adding einstein utility created in .local/bin to path
echo ""
echo "Adding einstein to PATH..."
echo ""
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

# Applying changes
echo ""
echo "Activating einstein shell utility command..."
echo ""
source ~/.bashrc

