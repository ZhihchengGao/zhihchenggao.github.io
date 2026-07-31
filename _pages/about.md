---
layout: about
title: about
permalink: /
subtitle: <span class="contact-email"><b>email:</b> richardgao (at) uchicago (dot) edu</span>

profile:
  align: right
  image: headshot.jpg
  image_circular: true # crops the image to make it circular
  more_info: >
    <p class="position-primary">Computer Science Undergraduate</p>
    <p class="position-affiliation">University of Chicago</p>
    <p class="position-primary">Research Assistant</p>
    <p class="position-affiliation">Virginia Image and Video Analysis Lab</p>

selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 5 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<style>
  .inline-affiliation img {
    height: 1em;
    width: auto;
    margin-right: 0.25em;
    vertical-align: -0.125em;
    display: inline-block;
  }
</style>

Hi, I'm **Jingbo Gao (高靖博)**. Most people call me **Richard** in the US.

I'm an undergraduate studying Computer Science at <a href="https://www.uchicago.edu" class="inline-affiliation"><img src="{{ '/assets/img/uchicago.png' | relative_url }}" alt="University of Chicago logo" loading="lazy" />University of Chicago</a>, currently working as a research assistant at the <a href="https://engineering.virginia.edu/labs-groups/virginia-image-and-video-analysis" class="inline-affiliation"><img src="{{ '/assets/img/uva.png' | relative_url }}" alt="University of Virginia logo" loading="lazy" />Virginia Image and Video Analysis (VIVA) Lab</a>. 


I'm in the early stages of exploring my directions in machine learning, particularly machine perception: To what extent can a machine process and learn from different modalities the way humans do, and how can this better inform a machine’s decisions? These questions have drawn me toward computer vision and multimodal machine learning as my current research interests.

I'd love to connect with mentors or peers thinking about similar questions, and I'm actively looking for research opportunities in CV and multimodal ML, in Chicago and beyond.

{% assign more_about_me = site.posts | where: "slug", "a-bit-more-about-me" | first %}
Here's [a bit more about me]({{ more_about_me.url | relative_url }}).
