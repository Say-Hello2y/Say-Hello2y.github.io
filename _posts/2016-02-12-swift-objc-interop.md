---
layout: post
title:  "Swift Objective-C Interoperation"
date:   2016-02-05 10:49:23 +0900
description: Does it work?
comments: true
tags: 
- swift 
- objc
---

So, you aren’t really sure if you should go full Swift on your million dollar app idea and is considering going half way. You tell yourself, “I can fill in the fact that Swift is a very young language with some heavy Objective-C gut that has been supported by the iOS community over the years.” You may be right but as you go down the road towards the million dollars, you might start to think otherwise.

## In the Beginning

In WWDC 2014 or 2013, Swift was released along with iOS 7 and one of the promises Apple made was that Swift will be great to work with Objective-C and that we would be able to do all the things in Swift that we could do in Objective-C. But can we???


Sorry if I made you throw up a little bit there. As you would know if you have already tried some interop, type briding between Swift and Objective-C requires you to check EVERYTHING. This is of course because Swift needs to know the type of an object at compile time and there is no such thing as nil (well, for most things)

So, in the end, these as?‘s will start to make you feel sick and make you think if that million dollars was just a little dream of yours. Apple felt kinda bad about this and released some ways of making things nicer with if-lets and guards but still, you can’t evade the truth.

Oh and by the way, as doesn’t really work when you’re go from Objective-C to a Swift protocol.

```swift
let seller = ObjcObject()
let items = seller.sellItems() as? [Cookie]
```

Here, Cookies is a Swift protocol and because Objective-C doesn’t know about it, the compiler immediately complains… So you have to do this

```swift
let seller = ObjcObject()
let items = seller.sellItems().map { $0 as Coookie }
```

and things turn out to be okay. But there are times when you return a generic NSArray and when you use an as, the array you get is actually not the type you specified. Thanks a lot, Chris Lattner.

## @obj everywhere

Have you had error messages that go “class needs to be Objective-C because it implements this stupid protocol”. You Google for a bit and throw in a @obj and once it compiles, you move on. But by writing those 4 characters, you just kinda screwed yourself and inhibited yourself from Swift awesome-tasticness.

## XCTests?

Oh, they don't work.

> rdar://24200114: Bi-directional interop between Objective-C and Swift does not work in XCTest targets.”

## So.. Should I?
I guess that depends on how your app has been structured. But in general, I think Apple’s done a good job getting things to at least work (except for XCTests)
