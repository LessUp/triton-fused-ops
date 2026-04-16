// Custom JavaScript for Triton Fused Ops
// ======================================

(function() {
  'use strict';

  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', function() {
    initCodeCopy();
    initSearchEnhancement();
    initExternalLinks();
    initSmoothScroll();
    initLazyLoading();
    initTableOfContents();
  });

  /**
   * Add copy button to code blocks
   */
  function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('div.highlighter-rouge');
    
    codeBlocks.forEach(function(block) {
      const button = document.createElement('button');
      button.className = 'btn-copy';
      button.setAttribute('aria-label', 'Copy code to clipboard');
      button.setAttribute('title', 'Copy');
      button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
      
      button.addEventListener('click', function() {
        const code = block.querySelector('code');
        if (code) {
          navigator.clipboard.writeText(code.textContent).then(function() {
            button.classList.add('copied');
            button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
            
            setTimeout(function() {
              button.classList.remove('copied');
              button.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
            }, 2000);
          });
        }
      });
      
      block.style.position = 'relative';
      block.appendChild(button);
    });
  }

  /**
   * Enhance search functionality
   */
  function initSearchEnhancement() {
    const searchInput = document.querySelector('.search-input');
    if (!searchInput) return;
    
    // Add keyboard shortcut (Cmd/Ctrl + K)
    document.addEventListener('keydown', function(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      }
    });
    
    // Track search events for analytics
    let searchTimeout;
    searchInput.addEventListener('input', function(e) {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(function() {
        const query = e.target.value.trim();
        if (query.length >= 3) {
          // Could send to analytics here
          console.log('Search query:', query);
        }
      }, 500);
    });
  }

  /**
   * Open external links in new tab
   */
  function initExternalLinks() {
    const externalLinks = document.querySelectorAll('a[href^="http"]:not([href*="' + window.location.hostname + '"])');
    
    externalLinks.forEach(function(link) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
      
      // Add external link indicator if not present
      if (!link.querySelector('.external-link-icon')) {
        const icon = document.createElement('span');
        icon.className = 'external-link-icon';
        icon.innerHTML = ' ↗';
        icon.style.fontSize = '0.75em';
        link.appendChild(icon);
      }
    });
  }

  /**
   * Smooth scroll for anchor links
   */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
      anchor.addEventListener('click', function(e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
          
          // Update URL without jumping
          history.pushState(null, null, targetId);
        }
      });
    });
  }

  /**
   * Lazy load images
   */
  function initLazyLoading() {
    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.classList.remove('lazy');
            img.classList.add('loaded');
            imageObserver.unobserve(img);
          }
        });
      });
      
      document.querySelectorAll('img[data-src]').forEach(function(img) {
        imageObserver.observe(img);
      });
    }
  }

  /**
   * Generate and enhance table of contents
   */
  function initTableOfContents() {
    const content = document.querySelector('.main-content');
    if (!content) return;
    
    const headings = content.querySelectorAll('h2, h3');
    if (headings.length < 3) return;
    
    // Check if TOC already exists
    if (document.querySelector('.toc')) return;
    
    const toc = document.createElement('nav');
    toc.className = 'toc';
    toc.setAttribute('aria-label', 'Table of contents');
    
    const tocTitle = document.createElement('h4');
    tocTitle.textContent = 'On this page';
    toc.appendChild(tocTitle);
    
    const tocList = document.createElement('ul');
    
    headings.forEach(function(heading) {
      if (!heading.id) {
        heading.id = heading.textContent.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
      }
      
      const li = document.createElement('li');
      li.className = 'toc-item toc-item-' + heading.tagName.toLowerCase();
      
      const a = document.createElement('a');
      a.href = '#' + heading.id;
      a.textContent = heading.textContent;
      a.addEventListener('click', function(e) {
        e.preventDefault();
        heading.scrollIntoView({ behavior: 'smooth' });
        history.pushState(null, null, '#' + heading.id);
      });
      
      li.appendChild(a);
      tocList.appendChild(li);
    });
    
    toc.appendChild(tocList);
    
    // Insert TOC at the beginning of content
    const firstHeading = content.querySelector('h1, h2');
    if (firstHeading) {
      firstHeading.parentNode.insertBefore(toc, firstHeading.nextSibling);
    }
    
    // Highlight current section on scroll
    let currentHeading = null;
    
    const headingObserver = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          const activeLink = toc.querySelector('a[href="#' + id + '"]');
          
          if (activeLink) {
            toc.querySelectorAll('a').forEach(function(link) {
              link.classList.remove('active');
            });
            activeLink.classList.add('active');
          }
        }
      });
    }, { rootMargin: '-100px 0px -60% 0px' });
    
    headings.forEach(function(heading) {
      headingObserver.observe(heading);
    });
  }

  /**
   * Add copy button styles dynamically
   */
  const copyButtonStyles = document.createElement('style');
  copyButtonStyles.textContent = `
    .btn-copy {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      padding: 0.5rem;
      background: rgba(30, 30, 30, 0.8);
      border: 1px solid #444;
      border-radius: 6px;
      color: #888;
      cursor: pointer;
      opacity: 0;
      transition: all 0.2s ease;
      z-index: 10;
    }
    
    div.highlighter-rouge:hover .btn-copy {
      opacity: 1;
    }
    
    .btn-copy:hover {
      background: #333;
      color: #fff;
      border-color: #666;
    }
    
    .btn-copy.copied {
      background: #28a745;
      border-color: #28a745;
      color: white;
    }
    
    .toc {
      background: rgba(114, 83, 237, 0.05);
      border-radius: 12px;
      padding: 1.25rem;
      margin: 1.5rem 0;
    }
    
    .toc h4 {
      margin-top: 0;
      margin-bottom: 0.75rem;
      font-size: 0.875rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #666;
    }
    
    .toc ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    
    .toc-item {
      margin: 0.375rem 0;
    }
    
    .toc-item-h3 {
      padding-left: 1rem;
    }
    
    .toc a {
      color: #666;
      text-decoration: none;
      font-size: 0.875rem;
      transition: color 0.2s;
    }
    
    .toc a:hover,
    .toc a.active {
      color: #7253ed;
    }
    
    .external-link-icon {
      opacity: 0.6;
    }
    
    img.lazy {
      opacity: 0;
      transition: opacity 0.3s;
    }
    
    img.loaded {
      opacity: 1;
    }
  `;
  document.head.appendChild(copyButtonStyles);

})();
