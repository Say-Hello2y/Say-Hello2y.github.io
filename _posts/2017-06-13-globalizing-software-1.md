---
layout: post
title:  "Globalizing Software"
date:   2017-06-13 15:49:23 +0900
description: UI shouldn't uniform for all countries.
comments: true
tags:
-  ui
---

**TL;DR** UI should not be the same for all countries for cultural reasons.

Here's a question you might want to ask yourself about your product.
> How is your product going to feel native to a user in a different country with an entirely different culture?

Before we try to answer this question, let's look at some of the existing methods most companies and software use to target the global audience

## Localization
To most developers, **going global means adding a localization feature on their app** so that a button with the text "Search" becomes "검색" in South Korea or "Chercher" in France.

Localization is a feature most commonly implemented as it is the easiest to do so. The only difference between applications is the level of localization.

For example, while both application A and B have all text within the app localized, application B could have the name of the application, the native settings, and the documentation localized - it's small, but the small things matter.

Check out Instagram's localized settings page.

Instagram (US)            |  Instagram (Korea)
:-------------------------:|:-------------------------:
![]({{ site.url }}/assets/global/instagram_english.png)  |  ![]({{ site.url }}/assets/global/instagram_korean.png)

## Basic Cultural Support
Now this is an approach that has been given slightly more thought than the basic localization and is something people should definitely start picking up as a must-have feature for their global apps.

These **basic cultural support features** take in consideration of various cultural conventions. Those conventions may include

* Writing system
* Directionality
* Native Font
* String Formats	
* Date
* Time
* Temeperature
* Currency
* Keyboard layouts

and so on...

Again, these are all very small features but you have no idea how frustrating it is to read Korean that is not in its own native font; 
and I'm gussing the same thing applys to all languages that are not based on the alphabet.

Applying these features allow the user to feel comfortable about the application's content and hence forth allow better interaction with it.

## But something is missing...
The two features mentioned above 
- Localization
- Cultural support

allows the app usable in various countries but there's something missing here.

Here's an analogy. 
It's almost as if an American car company tries to export a car to England after only having localized all the buttons in the car. No one is going to buy it in England because people in England drive on the wrong side of the road.

What I'm trying to say is that the **UI** also has to adapt to the target culture.

Check out the **[next blog post]({{ site.url }}{% post_url 2017-06-13-globalizing-software-2 %})** to find out how to do this.

