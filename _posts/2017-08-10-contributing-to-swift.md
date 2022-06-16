---
layout: post
title: "Contributing to Swift"
date: 2017-08-10 09:32:03+0900
description: "Getting started"
comments: true
tags:
- swift
- open-source
---

[🇰🇷 available]({{ site.baseurl }}{% post_url /kr/2017-08-10-contributing-swift-kr %})

If you have been using Swift for awhile, you may have heard about it being open source. And if you have been using Swift enthusiastically, you may have had some thoughts about wanting to contribute to the Swift project. You probably then went to [Swift's Github repo](https://github.com/apple/swift) and looked at couple of files just to get scared away by the huge codebase.

Well, I am here to tell you that **"Yes! You can contribute something to Swift."** I won't say that it's easy but I will say that it doable and very rewarding.

# Overview

In this two part blog post series, I will divide the process of contributing to Swift into 5 phases.

1. Identify the problem
2. Build the project
3. Implement the thing
4. Test the thing
5. Ship the thing

We will first cover parts 1 and 2 in this post.

Before we get started, I want to make something very clear.

> **If you get stuck, ask people on Twitter or send an email to swift-dev@swift.org. The Swift community will help you.**

# Step 1: Identifying the problem
For me, I decided to contribute to Swift when I found a simple bug while working in Xcode. It had to do with Swift's automatic fixit feature not providing a fixit for placing an `@escaping` keyword. Turns out, the Swift compiler provides a fixit when closure is a function parameter but does not provide one if the closure is a parameter of a returning function.

Here's the working version of the code.
```swift
func mapping <A, B, C> (f: @escaping (A) -> (B)) -> (@escaping ((C, B) -> (C))) -> (C, A) -> (C) {
    return { reducer in
        return { accum, input in
            reducer(accum, f(input))
        }
    }
}
```
> Curious what this does? Checkout my  **[Transducers in Swift]({{ site.url }}{% post_url 2017-07-17-transducers %})**  blog post

The compiler provides a fixit if the `@escaping` next to `f:` is emitted but *does not* provide one when the returning closure's paramter's `@escaping` is emitted.

## Asking the Swift community
At this point, I thought I had found a bug but had no idea what to do next. So I asked people in the Swift mailing list and got this reply.

```
This has been reported as SR-5556. 
In the future, please report bugs like this through bugs.swift.org and/or Radar.

Thanks,

~ Robert Widmann
```

You can see the bug [here](https://bugs.swift.org/browse/SR-5556).

## Getting the thumbs up 👍

When Robert said that the bug was filed on `bugs.swift.org`, I said to myself 

> "My job here is done. I did my part and I will be moving on now."

2 weeks went by and being curious if the bug had been patched, I went on the webiste to see what had happened. *Nothing…* Then a voice in my head started saying 

> "If you found it, why don't you fix it youself?"

So I left a comment on the wesite asking if I can take this bug as my beginner bug. Next day, I got a reply.

{: .center}
![JIRA comment]({{ site.url }}/assets/swift_contrib/jira_comment.png)

At this point, I was thinking to myself

> Uh oh. What have I done... 🤦‍♂

# Step 2: Building the Project

This step is pretty mechanical so just follow the steps Apple tells you to follow. [GO READ THIS](https://github.com/apple/swift/blob/master/README.md) but if you really don't want to, here's a summary; note that this summary only applies to macOS users.

1. Install `Homebrew`
  - <code>/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
2. `brew install cmake ninja`
3. `mkdir swift-source & cd swift-source`
4. `git clone https://github.com/apple/swift.git`
5. `./swift/utils/update-checkout --clone`
6. `sudo xcode-select -switch /Applications/Xcode_ver_num.app`
   - **GO READ** [system requirements](https://github.com/apple/swift/blob/master/README.md#system-requirements) and get the **CORRECT XCODE VERSION** before you do this
7. `./swift-src/swift/utils/build-script -x`
    - Build and produce an Xcode project for you to work in

{: .center}
![CAUTION! Hot!]({{ site.url }}/assets/swift_contrib/cpu.png)

Your computer will get really hot for about 2 hours so step away from it and go do something else.

After 2 episodes of GoT, an Xcode project will have been generated in `./swift-source/build/Xcode-DebugAssert/swift-macosx-x86_64`.

Open the Xcode project and you will see a familar inferface with a bunch of files. 

![Umm...]({{ site.url }}/assets/swift_contrib/xcode.png)

# Step 3: Implement the thing (To be continued...)
Check out the 👉👉 **[NEXT BLOG POST]({{ site.url }}{% post_url 2017-08-24-contributing-to-swift-2 %})** to see the last three remaining steps.

## ⚠️ Side Note

So… what actually happened was that the bug I described above turned out to be really really hard to fix. Fixing it would have involved some serious type system wizardry, something I don't have. Because of this, Slava Pestov, one of the engineers at the Swift comipler team, suggested for me to work on [this](https://bugs.swift.org/browse/SR-910) bug.

What's the lesson here? **Make sure to ask questions along the way if you get stuck** because the problem you are solving may be NP very very hard.
