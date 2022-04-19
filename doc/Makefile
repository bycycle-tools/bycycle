# Minimal makefile for Sphinx documentation
#   This make is adapted & extended from the original sphinx file

##########################################################################
## Settings

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = bycycle
SOURCEDIR     = .
BUILDDIR      = _build

# Custom settings
GITHUBORG     = https://github.com/bycycle-tools
STABLE        = 1.0.0
DEVELOPMENT   = ""
RECENT        = 0.1.3

##########################################################################
## Standard sphinx tasks (from sphinx)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

##########################################################################
## Custom tasks

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

# Build multiple versions of documentation
html-multi:
	# Clean out existing build
	$(eval current_branch := $(shell git branch -l | grep "\*" | sed "s/.* //" ))
	make clean
	# Create clean remote
	git remote remove upstream_temp || true
	git remote add upstream_temp https://github.com/bycycle-tools/bycycle.git
	git fetch upstream_temp --tags
	# Create a clean stable branch
	git branch -D v$(STABLE) || true
	git branch v$(STABLE) tags/$(STABLE)
	# Checkout the stable release tag
	git checkout v$(STABLE)
	# Create the stable site
	make html
	rm -rf _build/doctrees auto_examples auto_tutorials generated
	mv _build/* _build/html_stable
	git checkout $(current_branch)
	# Build develompent and recent sites
	$(eval BUILDS := $(DEVELOPMENT) $(RECENT))
	for v in $(BUILDS); do \
		if [ -z $$v ]; then continue; fi; \
		git clean -f; \
		git branch -D v$$v || true; \
		git branch v$$v tags/$$v; \
		git checkout -f v$$v; \
		if [ ! -f _templates/navbar.html ]; then \
            git checkout v$(STABLE) _templates/navbar.html -3; \
            git commit -m "navbar updated"; \
    	fi; \
		make html; \
		mkdir -p _build/html_stable/v$$v; \
		mv _build/html/* _build/html_stable/v$$v; \
		rm -rf auto_examples auto_tutorials generated _build/doctrees _build/html; \
		git checkout $(current_branch);\
		git branch -D v$$v; \
	done
	git branch -D v$(STABLE)
	# Move stable back to root reference
	mv _build/html_stable _build/html
	# Remove the temporary remote
	git remote remove upstream_temp

# Deploy the site to github pages site
deploy:
	# Clone, specifically, the gh-pages branch, putting it into '_build/gh_pages/'
	#   -b gh-pages --single-branch fetches specifically and only the gh-pages branch
	#   --no-checkout just fetches the root folder without content
	#   --depth 1 is a speed optimization since we don't need the history prior to the last commit
	git clone -b gh-pages --single-branch --no-checkout --depth 1 $(GITHUBORG)/$(SPHINXPROJ) _build/gh_pages

	# Add a .nojekyll file to tell Github pages to bypass Jekyll processing
	touch _build/gh_pages/.nojekyll

	# Copy site into the gh-pages branch folder, then push to Github to deploy
	cd _build/ && \
	cp -r html/* gh_pages && \
	cd gh_pages && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'deploy docsite' && \
	git push

# Clean & rebuild the html site, then deploy docsite
install:
	make clean
	make html
	make deploy

# Clean & rebuild the multi-version html site, then deploy docsite
install-multi:
	make clean
	make html-multi
	make deploy
