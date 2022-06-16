---
layout: post
title:  "Xcode Auto Generated Interfaces"
date:   2018-07-18 15:23:10 +0900
description: "Swift Objective-C Magic"
comments: true
tags: 
- swift
- objc
---

If you have ever called Objective-C code from Swift, you may have noticed the "Generated Interfaces" feature in Xcode. But have you noticed the details of the said "generated interfaces"?

For those of you who don't know what I'm talking about, Xcode has a feature where you can view the automatically generated Swift interfaces of Objective-C files from the editor. This is way, you can see how your Objective-C expressions are parsed into their Swift counterparts, **before compiling anything.**

{: .center}
![How to bring up Generated Interfaces]({{ site.url }}/assets/auto-gen-interface/howto.png)

## Look closely 👓

That's cool and all but did you notice how objc expressions were exactly parsed into their Swift counterparts? If you didn't, let me point them out to you.

## Initializers

```objc
- (instancetype)initWithName:(NSString *)name age:(NSInteger)age;
- (instancetype)initName:(NSString *)name;
- (instancetype)initFor:(NSInteger)years;
```

🔻 Generated Interfaces 

```swift
init(name: String, age: Int)
init(name: String)
init(for years: Int)
```

Notice that the `initWithName` is automatically shrunk to `init(name:)` to follow Swift's naming style 😱. This allows you to name your objc functions without worrying about how they are going to look in Swift.

Also, notice how `initFor` is separated into `init(for:)`. Cool, eh?

## Functions

Assuming the above initializers are for the `Animal` class, let's make our `Animal` do something by adding some functions.

```objc
- (NSInteger)sleepFor:(NSInteger)minutes; 
- (NSInteger)playAt:(Location)location for:(NSInteger)minutes; 
- (NSInteger)playWith:(Toy)toy                         
```

🔻 Generated Interfaces 

```swift
func sleep(for minutes: Int) -> Int
func play(at location: Location, for minutes: Int) -> Int
func play(with toy: Toy!) -> Int
```

Here, things seem pretty straightforward but once again, notice the details! The `sleepFor:`, `playAt:`, `playWith:` have been parsed as `sleep(for:)`, `play(at:)` and `play(with:)`.

## enums

Now that the `Animal`s can do something, let's make some `Location`s where we can play with our `Animal`s.

```objc
typedef NS_ENUM(NSInteger, Location) {
    LocationPark,
    LocationHome,
    LocationLake
};
```

Notice how I named the enum cases like how any good objc developer should; `TypeNameCaseName`.

That's cool but Swift favors simplicity and readability when it comes to naming. Thankfully, Xcode automatically takes care of that for you, again.

```swift
public enum Location : Int {
    case park
    case home
    case lake
}
```

Now look at the below Swift code that is calling objc code. Doesn't it look like you are calling native Swift code?

```swift
guard let 🦄 = Animal(name: "Shadowfax", age: 9000) else {
    fatalError("neighhhhh")
}
        
🦄.play(with: .boomerang)
🦄.play(at: .park, for: 10)
```

## Wrapping Up

Well, I hope you found that to be some what interesting. In the next post, I will maybe look into how this is done behind the scenes 😎

> Here's a [sample Xcode project](https://www.dropbox.com/sh/vjth76ug53mq5fa/AABn5ukKNuVszOXAEugy7mh_a?dl=0) for you to play around with.
