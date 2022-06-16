---
layout: post
title:  "Transducers in Swift"
date:   2017-07-17 10:17:11+0900
description: "Will it blend?"
comments: true
tags:
- functional-programming
- swift
---
# The necessary metaphor 🌯🌯🌯
Let's say that you work at a teddy bear factory. Your boss comes and tells you to remove all the faulty teddy bears from the line and package the good ones after putting a price tag on them.

Easy enough. You go to the conveyor belt where teddy bears are born and start looking at the teddy bears one by one. As you scan them, you throw away the faulty ones while labeling and boxing the good ones. If your quota for the day is 1000 bears, you would only have to scan 1000 bears.

**Now, what would happen if we write a program for this?**
In Swift, we could do something like this
```swift
let packagedBox = bears.filter(isValid).map(putPriceTag)
```

Cool, but let's see whats going on here. By doing `bears.filter(isValid)`, you are throwing away the faulty ones but also packaging the good ones into a box.

When the time comes for you to put a price tag on the good bears by doing `map(putPriceTag)`, you notice that you need to re-open the box and look at all the bears for the second time. At the end of the day, you feel twice as tired and you should because you just scanned twice as many teddy bears!

You end up getting home, feeling tired after looking at so many teddy bears and search for a solution on StackOverflow. And it turns out, there is an idea called 'Transducers' from Clojure that seems like a perfect solution to the problem. So you start Googling for answers...

{: .center}
![TED!](https://media.giphy.com/media/10lvrPfoXDHoTm/giphy.gif)

# What are Transducers?
> Transducers modify a process by transforming their internal reducing functions.

The basic purpose is to look again at `map` and `filter` and see if there is some idea in them that can be made more reusable.

We can do this by recasting them as process transformations;
or successions of steps that ingests an input and blots out an output.

If you think about it this way, `map` basically does what we said above and stores the collection of outputs into a collection.
That's a specialization of the idea. The generalized form of the idea is the "seeded left reduce"; taking something we are building up and a new thing and continuing to building up.

So, we want to get away from the idea that reduction is about creating a
praticular thing. Instead, we should focus more on it being a process because some processes build things while others are infinite.

The concept may be hard to understand at first - it certainly took me awhile - so, here are some gifs because gifs are good...

*Map*
![STL Map]({{ site.url }}/assets/transducers/map.gif)
*Transducers*
![Transducers]({{ site.url }}/assets/transducers/transducers.gif)

# Implementation
Before going all crazy, let's process lists in the naive/easy way.

## Naive Way
I'm guessing that from all the FP buzz, you are familiar with `map` and `filter`. So I'm going to use them to combine multiple functions to process an array of integers.

```swift
func isEven(_ x: Int) -> Bool {
  return x % 2 == 0
}

func incr(_ x: Int) -> Int {
  return x + 1
}

let naive1 = (1...10).map(incr)
                     .filter(isEven)

>> [2, 4, 6, 8, 10]
```

 ¯\_(ツ)_/¯

# Analysis
The above function calls `map` and `filter` twice on a range of integers.
And the performance-conscience you may say,

> "Hey, that just looped over that range n * 2 times and created an intermediate array! Can we make it so that it iterates only n times?"

And I would say
> "Yes, we can. We can use transducers for that"

But first, here's some theoretical stuff we need to cover first.

# Reduce everything
You need to understand that **all list processing functions - such as `map`, `filter` - can be redefined in terms of `reduce`**

But what does this have to do with transducers?

Recognizing this gives us regularity/uniformity because the things we can prove about `reduce` can also apply to the rest of the list processing functions as well.

Basically, if theory `A` applies to `reduce` and `map` can be expressed in terms of `reduce`, `A` must also apply to `map`.

Here are some examples to illustrate this.

```swift
func append<T>(to accum: [T], with input: T) -> [T] {
  return accum + [input]
}

extension Collection {
  typealias A = Iterator.Element

  func mmap<B>(_ f: @escaping (A) -> (B)) -> [B] {
      return reduce([]) { accum, elem in
          append(to: accum, with: f(elem))
      }
  }

  func mfilter(_ f: @escaping (A) -> (Bool)) -> [A] {
      return reduce([]) { accum, elem in
          if f(elem) {
              return append(to: accum, with: elem)
          } else {
              return accum
          }
      }
  }
}
```

Notice that `mmap` reduces with `append` and `mfilter` reduces with `append` as well. But it is important to recognize that **we chose to append `elem` to `accum` to create a new collection in both cases.**

```swift
let naive2 = (1...10).mmap(incr)
                     .mfilter(isEven)
naive1 == naive2
>> true
```

So, this works. But, if you haven't noticed yet, both `mmap` and `mfiler` still use intermediate arrays to process elements.
What this means is that everytime `mmap` prepares to process things, it **starts with an empty array** and so does `mfilter`.

This means that `naive2` still had to iterate n * 2 times.

# Transducers

We can do better than that. And this is where transducers come in.

> Transducers allow us to use only **one intermeidate array** and **one iteration through the array** to apply many transformations while being in control of the way it reduces.

### Thoughts before going in...

So, how should we go about this? Well right now, we know that both `mmap` and `mfilter` are implemented using `reduce` and that they both use a function called `append`. But why do we use `append` here? Do we even **have** to use it here? There's nothing special about it. Afterall, it's just a function 😃

If we think about it that way, I could use any function with the type `(accum, elem) -> (accum)` in place of `append`. Turns out, functions with the following type signitures are called **reducing functions**. Let's go ahead and write a version of map/filter that can take a reducing function in its closure.

# Code

```swift
func mapping <A, B, C> (f: @escaping (A) -> (B)) -> (@escaping ((C, B) -> (C))) -> (C, A) -> (C) {
  return { reducer in
      return { accum, input in
          reducer(accum, f(input))
      }
  }
}

func filtering<A, C> (f: @escaping (A) -> (Bool)) -> (@escaping ((C, A) -> (C))) -> (C, A) -> (C) {
  return { reducer in
      return { accum, input in
          if f(input) {
              return reducer(accum, input)
          } else {
              return accum
          }
      }
  }
}
```
😵😵😵😵

I know it looks like a lot but stay with me. It's not TOO complicated.
Here's what each parameter/generic type is trying to say.

 - `A` -> Input type
 - `B` -> Output type
 - `C` -> Accumulated data type (Array, Int, etc)
 - `f: (A) -> (B)`
   - Transformation function that takes an input and returns an ouput
 - `((C, B) -> (C)) -> ((C, A) -> (C))`
   - Returning closure parameters
     - *Input* : **Reducing function**. Takes `(accum, output)` and returns a new accumulated output.
     - *Output* : A closure fed into `reduce`. Takes a `(accum, input)` and applies `f()` to the input and calls `reducer`


# Example
To use these, let's start off with a simple reducing function; function that adds two numbers and returns a number.

```swift
func add(l: Int, r: Int) -> Int {
  return l + r
}

// or just (+)
```

In the first `reduce`, we **choose** to append the new element to the array
and in the second `reduce`, we **choose** to numerically add the new element to the initial value.

```swift
let clever = (1...10).reduce([], mapping(f: incr)(append))
                     .reduce(0, filtering(f: isEven)(+))

let oldWay = (1...10).map(incr).filter(isEven).reduce(0, +)

clever == oldWay
>> true
```
👍👍👍👍

In our old naive way, we have to add one more `reduce` after the `filter` to do the adding because we have no control over how these functions reduce.

To see what `mapping` and `filtering` does in detail, let's play with it.
As a reminder, they both return `(accum, input) -> (accum)` so we can feed it to `reduce`.

```swift
assert( mapping(f: incr)(append)([], 1) == [2] )
assert( mapping(f: incr)(append)([0,0], 1) == [0,0,2] )
assert( mapping(f: incr)(add)(0, 1) == 2 )

assert( filtering(f: isEven)(append)([1,1], 2) == [1,1,2] )
assert( filtering(f: isEven)(append)([1,1], 3) == [1,1] )
```

# The Big Moment.

Wait a sec... Did I just say `mapping` and `filtering` returns a `(accum, input) -> (accum)`?
Didn't I say that functions with that type signiture are called `reducing functions`?

So, `mapping` / `filterting` are the same things as `append` and `add`.
This is the moment we have been waiting for. Get excited...

Let's put a `filtering` where the reducing closure used to be.

```swift
//mapping(f: inr)(Any reducing functions here)
let incrAndFilterEvens: ([Int], Int) -> ([Int]) = mapping(f: incr)(filtering(f: isEven)(append))
let transducerRes = (1...20).reduce([], incrAndFilterEvens)

let oldRes = (1...20).map(incr).filter(isEven)
transducerRes == oldRes
>> true
```

As you see above, `incrAndFilterEvens` is also a reducing type.
This means we can keep on composing functions until we drop.

# Getting fancy 💃
This is cool but a bit messy. So let's make the process of combining functions a little more "pretty" by making it *functional*.

Here are some function composition operators.

```swift
// (f --> g)(x) = f(g(x))
infix operator --> : AdditionPrecedence
func --> <A, B> (x: (A), f: (A) -> (B)) -> (B) {
  return f(x)
}

// (f --> g)(x) = f(g(x))
func --> <A, B, C> (f: @escaping (A) -> (B), g: @escaping (B) -> C) -> (A) -> C {
  return { g(f($0)) }
}

let transduceFTW = (append) --> mapping(f: incr) --> filtering(f: isEven)
(1...10).reduce([], transduceFTW)
```

And if you print every time `append` is called, you get the following output.
```
[]
[2]
[2, 3]
[2, 3, 4]
[2, 3, 4, 5]
[2, 3, 4, 5, 6]
[2, 3, 4, 5, 6, 7]
[2, 3, 4, 5, 6, 7, 8]
[2, 3, 4, 5, 6, 7, 8, 9]
[2, 3, 4, 5, 6, 7, 8, 9, 10]
```

The array was created after going through the range only `n` times!

## 🎉🎉🎉
We managed to apply multiple transformations to an array whilst in full control of the reduction process! In addition, it looks pretty as hell. Here's a link to the [git repo containing the .playground file](https://github.com/mkchoi212/Transducers) if you want to play around with it.


*For some of you, Swift's `lazy` may have come into mind while reading this post. Don't worry as in the [next]({{ site.url }}{% post_url 2017-07-18-transducers-pt2 %}) post, we will look and compare transducer's performance and see how they can be used in the real world.*
