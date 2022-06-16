---
layout: post
title:  "Different ways to use Swift Extensions"
date:   2016-09-01 20:21:23 +0900
description: "Content may be controversial"
comments: true
tags: 
- swift 
- extensions
---

I've used them for awhile but I’m still not sure how I feel about them. In the beginning, I loved them and the new workflow they enabled. But as time went on, extensions became a go to garbage pile for code.

So, I'm going to try to organize my views on extensions by listing their various use cases and picking out the good ones.

## Private Helpers

In Objective-C, we had `.h` and `.m` files. And despite the fact that we had two files where we could just have one `.swift` file, there were some advantages. The biggest one was that we could just look at the .h file and look at all the externals. But at the same time, internal things were hidden away in the `.m` file; private properties/functions. So, how do we replicate this in Swift?

Let's first start off with a giant struct/class with all the internals — disregarding whether they are public or private.

We can refactor this by having a struct declared with all the public values and having a private extension to that struct with all the private values.

```swift
struct Pokemon {
   let name: String
   let health: Int
}
private extension Pokemon {
   mutating func apply(damage: Int){
   		health -= damage
   }
}
```

Now, no one can hurt our Pokemon with an evil intent!

## Grouping

I naturally came up with this awhile ago but the idea is that you can use extensions to just group various code blocks for your visual pleasure.

One could say **“Hey, that's not what extensions are for! Just use a `pragma mark` or `// Mark`” or “If you need to do this, you should probably start by refactoring your code anyways.”**

I would agree to both of those opinions but hey, that's a debate for 🍺 later.

The idea is simple. Let’s say you have a view controller and things are starting to look like it’s leaning towards the region of massive view controllers. So in order to tidy things up, you decide to break your code into chunks before you start with the real refactoring. You know you can use `// Mark` but you don’t like how they look. Well, you can use extensions instead.

```swift
extension MyMassivePokemonViewController {
   func addMorePokemons(group: [Pokemons]) {
   ...
   }
}
```

I agree this one is iffy. This is not what extensions were designed for but still, I think this could be useful for when you need to organize your large code base before starting to refactor to make things easier.

## Grouping for Protocol Conformance

I personally like this one the most. Tired of having code for `UITableViewDelegate` and `UITableViewDataSource` in the same place but don't want to create seperate files for them? Well, this one is for you.

Once again, the idea is simple. We are doing the same code grouping as we have done previously but only with code that conform to certain protocols.

```swift
extension MyTableViewControler : UITableViewDelegate {
    ...
}
extension MyTableViewControler : UITableViewDataSource{
    ...
}
```

This makes the seperation of code much easier to look at when compared to using pragma marks and is one of the most effective ways of using protocols — I think. Am I starting to sound like Scott Myers?

Oh and thanks to [NatashaTheRobot](https://www.natashatherobot.com) for most of these ideas!
