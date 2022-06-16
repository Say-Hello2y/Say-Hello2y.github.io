---
layout: post
title:  "Swift Sucess Enums"
date:   2016-02-05 10:49:23 +0900
description: Handle errors without optionals
comments: true
tags: 
- swift
---

**TL;DR** Enums may or may not be an optimal way to handle errors in Swift but are suitable to create types tailored to your needs.

Let’s talk about error handling in Swift. If you’ve been doing Swift for awhile, two things should pop into your head; optionals and try-catch blocks. They’ve served us well and will continue to do so, but let us look at an alternative method to handle errors.

## Why?

Swift’s optional types are great but has a single drawback; we don't retain any information regarding why the type became nil. This sucks because when it comes for us to report why an operation failed to the user, we have to say something vague like “We failed because of nil”

## How about an enum?

In order to capture our possible error, we want our failable function to return either the result or an ErrorType. So, why not return an enum instead of returning a “Optional(Result)” ?

With Swift’s enums ability to have associated values, we can do stuff like this to retain information about what went wrong.

```swift
// IMPLEMENTATION
enum LookupError: ErrorType {
  case InvalidName
  case NullData
}

enum UserResult {
  case Success(String)
  case Error(LookupError)
}
```

And then use it like this

```swift
func findUserStatus(name: String) -> UserResult {
  guard let userStats = users[name] else {
    return .Error(InvalidName)
  }
  return .Success(userStats)
}

switch findUserStatus("Stevie Wonder") {
  case let .Success(stats):
    print("Stevie Wonder's Stats: \(stats)")
  case let .Error(error):
    print("Error: \(error));
}
```

Here, we can see that Stevie’s information could not be found and has been properly reported to our user. This is much better than just doing an if let/guard on an optional and return a nil value when an error occurs.


## A Generic Version

But, this version doesn’t work if we want to return a `NSData`, does it? The Success case is only defined for a `String`. 

Let's use Swift’s generics to make it compatible with all types.

```swift
enum Result<T> {
  case Success(T)
  case Error(ErrorType)
}

// Example function signitures
func findUserAddress(name: String) -> Result<String>
func findUserAge(name: String) -> Result<Int>
```

##So?

If you look at the implementation of the Swift’s built-in error handling mechanisms, things work very similarly to what we just did with Result enums. But there is one limitation; error handling only works on the result type of a function. Swift won’t let you pass a possibly failed argument to a function but with our Result enums, you can because it’s just another value!

But in the end, most people probably prefer using optionals since they are built-in to the language and the syntactic sugars that come with it are convenient; like `??`. And so, here’s a quote I hope will help you make the decision.

> “Program into a language, not in it” - Steve McConnell

