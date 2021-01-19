# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = bycycle
SOURCEDIR     = .
BUILDDIR      = _build
STABLE        = 1.0.0
DEVELOPMENT   = ""
RECENT        = 0.1.3

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
	cp -r html/* gh_pages && \
	cd gh_pages && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push

# Build multiple versions
html-multi:
	# Clean out existing build
	$(eval current_branch := $(shell git branch -l | grep "\*" | sed "s/.* //" ))
	make clean
	# Create clean remote
	git remote remove upstream_temp || true
	git remote add upstream_temp https://github.com/bycycle-tools/bycycle.git
	git fetch upstream_temp --tags
	# Create clean branches
	git branch -D v$(STABLE) || true
	git branch v$(STABLE) tags/$(STABLE)
	# Checkout the stable release tag
	git checkout v$(STABLE)
	# Create the stable site
	make html
	rm -rf _build/doctrees auto_examples auto_tutorials generated
	mv _build/* _build/html_stable
	git checkout $(current_branch)
	git branch -D v$(STABLE)
	# Build develompent and recent sites
	$(eval BUILDS := $(DEVELOPMENT) $(RECENT))
	for v in $(BUILDS); do \
		if [ -z $$v ]; then continue; fi; \
		git branch -D v$$v || true; \
		git branch v$$v tags/$$v; \
		git checkout v$$v; \
		make html; \
		mkdir -p _build/html_stable/v$$v; \
		mv _build/html/* _build/html_stable/v$$v; \
		rm -rf auto_examples auto_tutorials generated _build/doctrees _build/html; \
		git checkout $(current_branch);\
		git branch -D v$$v; \
	done
	# Move stable back to root reference
	mv _build/html_stable _build/html
	# Remove the temporary remote
	git remote remove upstream_temp

# Build the multi-version html site, and push it to gh-pages branch of repo to deploy
install-multi:
	# Clean out existing build
	make clean
	# Build the sphinx site
	make html-multi
	# Clone, specifically, the gh-pages branch, putting it into '_build/gh_pages/'
	#   --no-checkout just fetches the root folder without content
	#   --depth 1 is a speed optimization since we don't need the history prior to the last commit
	#   -b gh-pages fetches only the branch for the gh-pages
	git clone -b gh-pages --single-branch --no-checkout --depth 1 https://github.com/bycycle-tools/bycycle _build/gh_pages
	# A .nojekyll file tells Github pages to bypass Jekyll processing
	touch _build/gh_pages/.nojekyll
	# Copy site into the gh-pages branch folder, then push to Github to deploy
	cd _build && \
	cp -r html/* gh_pages && \
	cd gh_pages && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push
