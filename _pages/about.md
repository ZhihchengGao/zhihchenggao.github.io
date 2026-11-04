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

Hey! I'm **Jingbo Gao (高靖博)**. Most people call me **Richard** in the US.

I'm an undergraduate studying Computer Science at <a href="https://www.uchicago.edu" class="inline-affiliation"><img src="{{ '/assets/img/uchicago.png' | relative_url }}" alt="University of Chicago logo" loading="lazy" />University of Chicago</a>, currently working as a research assistant at the <a href="https://engineering.virginia.edu/labs-groups/virginia-image-and-video-analysis" class="inline-affiliation"><img src="{{ '/assets/img/uva.png' | relative_url }}" alt="University of Virginia logo" loading="lazy" />Virginia Image and Video Analysis (VIVA) Lab</a>. I'm in the early stages of exploring my directions in machine learning. My main research interest lies in computer vision, machine perception, and multimodal machine learning. 

Beyond research, I am fascinated by the way people build and scale meaningful products. I want to see this process up close, which is why I am also interested in opportunities to work at a startup.

<!-- , particularly machine perception: To what extent can a machine process and learn from different modalities of the physical world the way humans do, and how can this better inform a machine’s decisions? These questions have drawn me toward computer vision, machine perception, and multimodal machine learning as my current research interests. -->

**I am actively looking for new opportunities!** I'm especially interested in:
- Research opportunities, especially in computer vision, multimodal ML, machine perception, robotics.
- Any technical intern positions (SWE, MLE, research, etc.) at startups!

{% assign more_about_me = site.posts | where: "slug", "a-bit-more-about-me" | first %}
Here's [a bit more about me]({{ more_about_me.url | relative_url }}).
