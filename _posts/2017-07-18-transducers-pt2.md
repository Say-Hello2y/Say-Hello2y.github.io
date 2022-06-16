---
layout: "post"
title: "Transducers in Swift II"
date: "2017-07-18 10:13"
description: "Lazy performance evaluation"
comments: true
tags:
- functional-programming
- swift
---

If you haven't read the first part of this post, I recommend you to [read it here first]( {{ site.url }}{% post_url 2017-07-17-transducers %} ) to see how transducers work. If you don't care about that kind of stuff, read on.

# Performance
So, transducers look cool and all but.. is it fast? To see how fast it is, I will compare various implementations of transducers with `map`, `filter`, and `reduce` that are provided by the Swift Standard Library. And let's throw in an imperative version of the loop into the list

## Setup
We will use [Attabench](https://github.com/lorentey/Attabench), an awesome benchmarking app made by [@lorentey](https://twitter.com/lorentey) to generate all the pretty graphs.

The tests will be based on three operations that will start with an array of random integers and

1. Increment by 1
2. Filter evens
3. Reduce by adding to 0

## Original Transducer vs. STL
```swift
// STL
let _ = input.map(incr).filter(isEven).reduce(0, (+))

// Transducer
let _  = input.lazy.reduce(0, (+) ==> mapping(f: incr) ==> filtering(f: isEven))
```

![Transducer (comp) vs. STL]({{ site.url }}/assets/transducers/comp.png)

Here, we can clearly see that the STL's version is faster. *But because we are using custom operators to chain operations, maybe the compiler isn't able to do aggressive optimizations.* Let's try this again without the custom operators.

## Transduers (no custom operators) vs. STL
```swift
// STL
let _ = input.map(incr).filter(isEven).reduce(0, (+))

// Transducer
let _  = input.reduce(0, mapping(f: incr)(filtering (f: isEven)(+)))
```

![Transducer (no comp) vs. STL]({{ site.url }}/assets/transducers/without_comp.png)

Well, it seems like using operators to make things pretty cost quite a lot of performance. Without that extra overhead, we can see that our transducers beat Apple's engineers!! 🎉🎉🎉

## But did we???
Turns out Apple engineers thought about the problem of intermediate arrays and made this thing called [LazyCollection](https://developer.apple.com/documentation/swift/lazycollection). It's the same old collection but implemented lazily.

Here is Apple's explanation of `LazyCollection`'s `lazy` property.

> Use the lazy property when chaining operations to prevent intermediate operations from allocating storage, or when you only need a part of the final collection to avoid unnecessary computation.

So... it does the same thing transducers do but with four letters...

{: .center}
![Welp](https://media.giphy.com/media/tpwwhv1BLd31e/giphy.gif)

Anyways, let's go ahead and apply `lazy` to both STL and transducer versions of the code to see what happens.

```swift
// STL (lazy)
let _ = input.lazy.map(incr).filter(isEven).reduce(0, (+))

// Transducer (lazy)
let _  = input.lazy.reduce(0, mapping(f: incr)(filtering (f: isEven)(+)))
```
![Lazy]({{ site.url }}/assets/transducers/lazy.png)

Well, it looks like our implementation performed about the same as Apple's implementation. But when tested out on an iPhone 7, **heap allocation of our version was much higher than the STL's; transducers allocated x10 memory in the heap... 😞😞😞**

While at it, let's benchmark an imperative version.

```swift
// EDITED 2017.07.21
// Return from @inline(never) because compiler is very aggressive about
// disregarding results that are not used

@inline(never) func loop(input: [Int]) -> Int {
    var res = 0
    for i in input where i % 2 == 0 {
        res += (i + 1)
    }
    return res
}
```

![Imperative]({{ site.url }}/assets/transducers/imperative.png)

Turns out `lazy` is as speedy as an imperative loop. 

{: .center}
![Right??](https://media.giphy.com/media/XreQmk7ETCak0/giphy.gif)

# So What?
> **Transducers may be as fast as Swift's `LazyCollection` but is highly inefficient.**

So when trying to come up with practical use cases for transducers...

<blockquote class="twitter-tweet tw-align-center" data-lang="en"><p lang="en" dir="ltr">I think transducers are a cool idea but unnecessary in Swift (because we have lazy). Seems like forcing FP when it&#39;s not necessary :)</p>&mdash; Chris Eidhof (@chriseidhof) <a href="https://twitter.com/chriseidhof/status/885441861699203072">July 13, 2017</a></blockquote> <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

## Lessons
1. **Don't force functional programming concepts into Swift.** Swift is not a purely functional language. It's a multi-paradigm language

2. **Grab a complicated functional programming language concept and try to implement it in your favorite programming language** if you want a real taste of FP

3. Write more Swift code 😃

Although transducers may not have much practicality, I feel like I took my first REAL steps into the world of functional programming. Reading "Little/Seasoned Schemer" didn't leave me feeling like a true functional programmer but aching over how to implement transducers certainly did!
