---
layout: post
title:  "Things we can learn from Microsoft"
date:   2017-12-23 15:22:11 +0900
description: from its early days
comments: true
tags:
- type-system
- algorithms
---

Let me tell you something about me. I was [obsessed with HackerNews](https://deadbeef.me/2017/12/rss) for about three months. Not a single day went by where I didn't check the front-page. 

Anyways, during those three months, hundreds of articles went past my fingers. I admit I don't remember all of them but there were few articles that lingered in the back of my head. And surprisingly, a big chunk of those articles had to do with Microsoft during its early days. So, this is me writing them down to see what I can learn from them.



So… here are **three lessons we can all learn from Microsoft during the 1980's**.



## Lesson #1. Optimize the most common operations

Before you quote Don Knuth, here is a story of how Microsoft Word became the most popular word processor in the world.



In 1983, PC Word 1.0 and Mac Word 1.0 shipped with a feature called "Piece Table". The piece table wasn't an official feature that was printed on the floppy disk but it was the data structure that enabled many of the listed features.



For example, it allowed for super fast copy and paste, undo and redo operations. You can go [here](https://web.archive.org/web/20160308183811/http://1017.songtrellisopml.com/whatsbeenwroughtusingpiecetables) to see how they were actually implemented but here is the TL;DR version of it.

> Instead of storing text in the document as a single long string of characters, one can maintain a small set of records; a piece table. This table can then hold a collection of data that is only a few bytes long - a piece - that describes how a string is fragmented into pieces.



Although this sounds quite mundane now, super fast undo/redo and paste operations were like black magic to people in the 80's. This is because back then, you had to go grab a cup of coffee and take a stroll around the park in your Reeboks while waiting for your word processor to finish copying and pasting a paragraph from the "American Psycho". Nevertheless, people loved these features they allowed them to write without being interrupted and therefore save countless hours.



Later when Word 3.0 launched, Microsoft revolutionized the word processing industry once again with a feature called fast save. I won't describe how they managed to implement [fast save](https://web.archive.org/web/20160308183811/http://1017.songtrellisopml.com/whatsbeenwroughtusingpiecetables) but I think that this feature is the best thing Microsoft has ever shipped… ever.



## Lesson #2. Make it backwards compatible

Microsoft has shipped many versions of their Windows operating system but here's a lesson we can all learn from them. 

> It takes only one program to sour an upgrade.

Now, allow me to elaborate.



Let's say you are an IT manager of a company and you find out that if you upgrade to a new fancy version of the OS, program X the entire company uses won't work anymore. You really want that one feature that comes with the new OS so you desperately make the call to the company that makes program X to request for an upgrade. But it turns out, the upgrade won't be free. At this point, would you upgrade to the new OS? 



Microsoft's Setup/Upgrade team figured out that every single user had a "deal-breaker" program, a program that must run or they won't upgrade. This is why Microsoft Windows has that one mysterious directory `C:\WINDOWS\AppPatch` that stores all the `.dll`s to support application backwards compatibility; this is why you can still play Sims 1 on your computer.



## Lesson #3. A.B.C. Always Be Considerate

![](https://media.giphy.com/media/nAZ3JTRUYiis0/giphy.gif)

Excuse my poor attempt to reference Glengarry Glen Ross but the message still stands; Always Be Considerate! Unlike Alec Baldwin here, I am not saying you should be considerate to the people you are trying to sell houses to. Instead, you should be considerate to the end users of your software.



When Microsoft wanted to make a scripting language for Excel users, they started a project called Visual Basic for Applications. In the beginning, the developers had to decide whether if they wanted the language to be *statically (strong) typed* or *dynamically (weak) typed*. Personally, I am in the statically typed languages camp as I love using Swift and Golang. These [two](http://blog.cleancoder.com/uncle-bob/2017/01/11/TheDarkPath.html) [posts](http://elbenshira.com/blog/the-end-of-dynamic-languages/) do a better job than me describing each side of the argument so I highly recommend you to read those when you have the time.

>  TL;DR The main gist of the argument is that statically typed languages allow you to find errors at compile time while dynamically typed languages allow you to find errors during runtime.

We could go on for days arguing over which one is better than the other and keep furiously typing away on our keyboards until our butterfly mechanism keys break. **However, the end-user, who is most likely to be an accountant at a paper company like Dunder Mifflin, won't care if it's statically typed or dynamically typed.** What that accountant cares about the most is whether if he can easily whip up a script for Excel that will let him automate repetitive tasks so he can go home to his kids a bit sooner.



If you decide to make the language statically typed, the intern that is in charge of writing the manual will have to describe to the accountant what a variable is and most importantly, what types are and why they are so important. Or you could make the language dynamically typed and let the accountant just start coding away immediately. Yes, the accountant will get runtime errors but he probably isn't writing a program for the Apollo 11 lunar module. All the accountant wants to do is add up column A through Z. 



You may have strong beliefs about a certain decision that has to be made in a project. However, are you being considerate of the user?
