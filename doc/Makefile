# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = bycycle
SOURCEDIR     = .
BUILDDIR      = _build
STABLE 		  = v0.1.3
DEVELOPMENT   = v1.0.0rc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# # Custom cleaner that also removes the generated files from sphinx-gallery
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf auto_examples
	rm -rf auto_tutorials
	rm -rf generated

# Check that builds site with nitpicky checking, and also does post-hoc link check
check:
	make SPHINXOPTS="-n" html
	make linkcheck

# Build the html site, and push it to gh-pages branch of repo to deploy
install:
	# Clean out existing build
	make clean
	# Clone, specifically, the gh-pages branch, putting it into '_build/gh_pages/'
	#   --no-checkout just fetches the root folder without content
	#   --depth 1 is a speed optimization since we don't need the history prior to the last commit
	#   -b gh-pages fetches only the branch for the gh-pages
	git clone -b gh-pages --single-branch --no-checkout --depth 1 https://github.com/bycycle-tools/bycycle _build/gh_pages
	# A .nojekyll file tells Github pages to bypass Jekyll processing
	touch _build/gh_pages/.nojekyll
	# Build the sphinx site
	make html
	# Copy site into the gh-pages branch folder, then push to Github to deploy
	cd _build/ && \
	cp -r html/ gh_pages && \
	cd gh_pages && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push

# Build multiple versions
html-multi:
	# Clean out existing build
	make clean
	# Ensure navbar template exists
	orig_branch=`git branch --show-current`
	git checkout $(STABLE)
	if [ ! -f _templates/navbar.html ]; then \
		git checkout $(DEVELOPMENT) _templates/navbar.html -3; \
		git commit -m "navbar updated"; \
	fi
	# Create the stable site
	make html
	# Cleanup stable build
	rm -rf _build/doctrees
	rm -rf auto_examples
	rm -rf auto_tutorials
	rm -rf generated
	mv _build/* _build/html_stable
	# Add versions to build here
	for v in $(DEVELOPMENT); do \
		git checkout $$v; \
		make html; \
		mkdir -p _build/html_stable/$$v; \
		mv _build/html/* _build/html_stable/$$v; \
		rm -rf auto_examples; \
		rm -rf auto_tutorials; \
		rm -rf generated; \
		rm -rf _build/doctrees; \
		rm -rf _build/html; \
	done
	# Move stable back to root reference
	mv _build/html_stable _build/html
	mkdir _build/html/$(STABLE)
	cp _build/html/index.html _build/html/$(STABLE)
