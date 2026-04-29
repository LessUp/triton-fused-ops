(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    localizeUi();
    initLanguageSwitcher();
    initPageOutline();
    initCodeCopy();
    initExternalLinks();
  });

  function isChinesePage() {
    return window.location.pathname.indexOf("/docs/zh/") !== -1;
  }

  function localizeUi() {
    var zh = isChinesePage();
    var backToTop = document.querySelector(".back-to-top-link");
    if (backToTop) {
      backToTop.textContent = zh ? "返回顶部" : "Back to top";
    }

    var searchInput = document.querySelector(".search-input");
    if (searchInput && !searchInput.getAttribute("data-localized")) {
      searchInput.placeholder = zh ? "搜索知识点" : "Search knowledge";
      searchInput.setAttribute("aria-label", zh ? "搜索知识点" : "Search knowledge");
      searchInput.setAttribute("data-localized", "true");
    }
  }

  function initLanguageSwitcher() {
    var path = window.location.pathname;
    var container = document.querySelector(".main-content");
    var heading = container && container.querySelector("h1");

    if (!container || !heading) {
      return;
    }

    var match = path.match(/\/docs\/(en|zh)\/(.*)/);
    if (!match) {
      return;
    }

    if (document.querySelector(".language-switcher")) {
      return;
    }

    var lang = match[1];
    var suffix = match[2] || "";
    var otherLang = lang === "en" ? "zh" : "en";
    var otherPath = path.replace("/docs/" + lang + "/", "/docs/" + otherLang + "/");
    var zh = lang === "zh";

    var wrapper = document.createElement("div");
    wrapper.className = "language-switcher";

    var label = document.createElement("div");
    label.className = "language-switcher-label";
    label.textContent = zh ? "语言切换" : "Language";

    var links = document.createElement("div");
    links.className = "language-switcher-links";

    links.appendChild(createLangLink(lang === "en" ? path : otherPath, "English", lang === "en"));
    links.appendChild(createLangLink(lang === "zh" ? path : otherPath, "中文", lang === "zh"));

    wrapper.appendChild(label);
    wrapper.appendChild(links);
    heading.insertAdjacentElement("afterend", wrapper);
  }

  function createLangLink(href, text, active) {
    var link = document.createElement("a");
    link.className = "language-switcher-link" + (active ? " is-active" : "");
    link.href = href;
    link.textContent = text;
    return link;
  }

  function initPageOutline() {
    var content = document.querySelector(".main-content");
    if (!content || document.querySelector(".page-outline")) {
      return;
    }

    var headings = Array.prototype.slice.call(content.querySelectorAll("h2, h3"));
    headings = headings.filter(function (heading) {
      return heading.textContent.trim().length > 0;
    });

    if (headings.length < 2) {
      return;
    }

    headings.forEach(function (heading) {
      if (!heading.id) {
        heading.id = slugify(heading.textContent);
      }
    });

    var outline = document.createElement("nav");
    outline.className = "page-outline";
    outline.setAttribute("aria-label", isChinesePage() ? "页内导航" : "Page outline");

    var title = document.createElement("div");
    title.className = "page-outline-title";
    title.textContent = isChinesePage() ? "本页内容" : "On this page";
    outline.appendChild(title);

    var list = document.createElement("ul");
    list.className = "page-outline-list";
    outline.appendChild(list);

    headings.forEach(function (heading) {
      var item = document.createElement("li");
      item.className = "page-outline-item depth-" + heading.tagName.toLowerCase().replace("h", "");

      var link = document.createElement("a");
      link.className = "page-outline-link";
      link.href = "#" + heading.id;
      link.textContent = heading.textContent;

      item.appendChild(link);
      list.appendChild(item);
    });

    var anchor = content.querySelector(".language-switcher") || content.querySelector("h1");
    if (anchor) {
      anchor.insertAdjacentElement("afterend", outline);
    }

    var links = Array.prototype.slice.call(outline.querySelectorAll(".page-outline-link"));
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) {
            return;
          }

          var activeId = entry.target.id;
          links.forEach(function (link) {
            link.classList.toggle("is-active", link.getAttribute("href") === "#" + activeId);
          });
        });
      },
      { rootMargin: "-20% 0px -65% 0px" }
    );

    headings.forEach(function (heading) {
      observer.observe(heading);
    });
  }

  function initCodeCopy() {
    var blocks = document.querySelectorAll("div.highlighter-rouge");
    if (!blocks.length) {
      return;
    }

    blocks.forEach(function (block) {
      if (block.querySelector(".copy-code-button")) {
        return;
      }

      var code = block.querySelector("code");
      if (!code) {
        return;
      }

      var button = document.createElement("button");
      button.type = "button";
      button.className = "copy-code-button";
      button.textContent = isChinesePage() ? "复制" : "Copy";
      button.setAttribute("aria-label", isChinesePage() ? "复制代码" : "Copy code");

      button.addEventListener("click", function () {
        navigator.clipboard.writeText(code.textContent).then(function () {
          button.classList.add("is-copied");
          button.textContent = isChinesePage() ? "已复制" : "Copied";
          window.setTimeout(function () {
            button.classList.remove("is-copied");
            button.textContent = isChinesePage() ? "复制" : "Copy";
          }, 1600);
        });
      });

      block.appendChild(button);
    });
  }

  function initExternalLinks() {
    var links = document.querySelectorAll("a[href^='http']");
    links.forEach(function (link) {
      if (link.hostname === window.location.hostname) {
        return;
      }
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    });
  }

  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[\s/]+/g, "-")
      .replace(/[^\w\u4e00-\u9fff-]+/g, "")
      .replace(/-+/g, "-");
  }
})();
