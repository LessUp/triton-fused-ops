source "https://rubygems.org"

gem "jekyll", "~> 4.3.3"
gem "bundler", "~> 2.4"

# Jekyll plugins
group :jekyll_plugins do
  gem "jekyll-remote-theme"
  gem "jekyll-seo-tag", "~> 2.8"
  gem "jekyll-sitemap"
  gem "jekyll-feed"
  gem "jekyll-redirect-from"
  gem "jekyll-minifier"
  gem "jekyll-include-cache"
end

# Windows and JRuby does not include zoneinfo files
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# HTML validation for testing
group :test do
  gem "html-proofer", "~> 5.0"
end
