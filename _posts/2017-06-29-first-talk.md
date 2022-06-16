---
layout: post
title:  "🔈 Swift Talk: Limiting the use of Protocols"
date:   2017-06-29 13:03:23 +0900
description: My first Swift talk
comments: true
tags: 
- swift 
- talk
---

**TL;DR** Practice like it's real. So when it's real, it's just like practice

# Product first

<div id="fb-root"></div>
<script>(function(d, s, id) {
  var js, fjs = d.getElementsByTagName(s)[0];
  if (d.getElementById(id)) return;
  js = d.createElement(s); js.id = id;
  js.src = "//connect.facebook.net/en_US/sdk.js#xfbml=1&version=v2.10&appId=84797675741";
  fjs.parentNode.insertBefore(js, fjs);
}(document, 'script', 'facebook-jssdk'));</script>


<center><div class="fb-post" data-href="https://www.facebook.com/realmkr/posts/1225888860872574" data-width="500" data-show-text="true"><blockquote cite="https://www.facebook.com/realmkr/posts/1225888860872574" class="fb-xfbml-parse-ignore"><p>Swift&#xb2c8;&#xae4c; &#xbb34;&#xc870;&#xac74; &#xd504;&#xb85c;&#xd1a0;&#xcf5c;&#xc744; &#xc368;&#xc57c; &#xd560;&#xae4c;&#xc694;? &#x1f914;
&#xd504;&#xb85c;&#xd1a0;&#xcf5c;&#xc744; &#xc81c;&#xb300;&#xb85c; &#xc0ac;&#xc6a9;&#xd558;&#xb294; &#xbc29;&#xbc95;&#xc744; &#xc54c;&#xb824;&#xb4dc;&#xb9bd;&#xb2c8;&#xb2e4;!</p><a href="https://www.facebook.com/realmkr/">Realm Korea</a>에 의해 게시 됨&nbsp;<a href="https://www.facebook.com/realmkr/posts/1225888860872574">2017년 7월 24일 월요일</a></blockquote></div></center>


![]({{ site.url }}/assets/let_us_go/talk_1.jpg)
![]({{ site.url }}/assets/let_us_go/talk_2.jpg)
![]({{ site.url }}/assets/let_us_go/talk_3.jpg)
![]({{ site.url }}/assets/let_us_go/talk_4.jpg)
*Credits to these awesome photos go to* [@gbmksquare](https://twitter.com/gbmKSquare)

# The Talk
The topic of the talk was...
> Using protocols everywhere in Swift is not a good thing.

## Protocols?
To see where I am trying to go with this, we first need to know what a protocol
is.
There are tons of tutorials online about protocols so I will keep it simple.

Object Oriented                       |  ✅ Protocol Oriented
:-----------------------------------:|:--------------------------------------:
![](http://machinethink.net/images/mixins-and-traits-in-swift-2/ShootingHelper.png) | ![](http://machinethink.net/images/mixins-and-traits-in-swift-2/GameTraits.png)

Basically, protocols allow you to go from the left to the right and that's
considered a good thing. 

Flatter architecture made possible with the concept of **composition is easier to deal with than a complex class hiearchy**.

## What's so bad then?
Let's look at some examples.

```swift
protocol URLStringConvertible {
    var urlString: String { get }
}

// ...

func sendRequest(urlString: URLStringConvertible, method: () -> ()) {
    let string = urlString.urlString
}
```

This `URLStringConvertible` is not accomplishing anything here and can simply be replaced by a value. 

But somehow, I think a lot people feel as if using protocols for everything is the right thing to do in Swift.
Maybe this is because of the large number of *"Protocol Oriented X"* tutorials on the web or maybe because Apple has been trying to sell Swift as a "Protocol Oriented Language". But whatever the reason,

> Using protocols without thinking about the consequences is **NOT** ideal

# More Examples
Suppose we are making a library that is good at creating `UIView` elements with data inserted in them. 
I'm going to completely ignore what I just said above and make a protocol first because that's the *"cool thing to do"*.

## Protocol Oriented Approach 

```swift
protocol HeaderViewProtocol {
    associatedtype Content
    func setHeader(data: Content)
}
```
That looks like a cool protocol that will make us look like we know what we are doing. 
Let's now apply this protocol to various UIView subclassess to build up our library.

```swift
class MyLabel: UILabel, HeaderViewProtocol {
    func setHeader(data: String) {
        self.text = data
    }
}

class MyButton: UIButton, HeaderViewProtocol {
    func setHeader(data: String) {
        self.titleLabel?.text = data
    }
}
```

Simple enough. I just successfully abstracted the idea of a class that can be used as a `HeaderView` with a single protocol.
Now, I want to make an array of `HeaderViewProtocol` elements so that I can later insert them into a `UICollectionView`.

```swift
let elements = [MyLabel(), MyButton(), UIStackView()]
```

Wait.. `UIStackView` isn't a `HeaderViewProtocol` but why does the compiler not raise any errors?
If you look at `elements` type, the Swift compiler tells us that it's just a `[UIView]` and not an array of `UIView`s that conform to `HeaderViewProtocol`.
You could go ahead and try things like

```swift
let elements : [HeaderViewProtocol] = [MyLabel(), MyButton(), UIStackView()]
let elements : [UIView<HeaderViewProtocol>] = [MyLabel(), MyButton(), UIStackView()]
```

but nothing works because there is not a way to express a class that conforms to a protocol type in Swift 3.
We could have done `UIView<HeaderViewProtocol>` in Objective-C but that's a whole different story.

> **Note**: turns out you can do this in **Swift 4** with class subtype existentials but this doesn't really change anything
```swift
let elements: [HeaderViewProtocol & UIView]
```

So in Swift 3, we need to go ahead and create a type eraser like this.

```swift
struct AnyHeaderView<Content>: HeaderViewProtocol {
    var _setHeader: (Content) -> ()
    
    init<T: HeaderViewProtocol>(_ view: T) where Content == T.Content {
        _setHeader = view.setHeader
    }
    
    func setHeader(data: Content) {
        return _setHeader(data)
    }
}
```

Ok, that's a lot of code to keep the compiler happy. Anyway, we can now go ahead make the array with type safety we want.
```swift
let elements = [AnyHeaderView(MyLabel()), AnyHeaderView(MyButton())]
```

## Value Oriented Approach
> But instead of **passing in a type that promises things, couldn't we just pass the things we promised?**

Read that again and think about that for a second.

And to do just that, let's make a type that can wrap all the things we promised into a struct.

```swift
struct HeaderView<T>{
    let view: UIView
    let setHeader: (T) -> ()
}
```

Now, we can do this...

```swift
let label = UILabel()
let labelHeader = HeaderView<String>(view: label) { str in
    label.text = str
}

let imageView = UIImageView()
let imageHeader = HeaderView<UIImage>(view: imageView ) { img in
    imageView.image = img
}
```
Using a struct instead of a protocol halved the code size. Once implemented, the solution seems almost too simple to be true that we wonder, "Why couldn't we think of this the first time?"
The reason maybe because we came up with a solution to a problem we didn't even try tounderstand.

> Pick the right tool for the job, not the other way around.

# Conclusion
If using protocols in your code is making you write unncessary code to keep the compiler quiet and satisfied, maybe you should consider using struct / function values instead.

Here's a general rule of thumb.

| Situation                                        | Solution |
|--------------------------------------------------|----------|
| 1 function in protocol?                          | Function |
| 1 > functions?                                   | Protocol |
| Used only once? (completion handlers, callbacks) | Function |
| Used a lot? (data source, delegate)              | Protocol |

# About the talk

This was my first ever programming talk. My content was very techinal and the talk had some live coding in the middle. So naturally, I was nervous.

The talk was held in Korea, at one of [Kakao's buildings](http://www.kakao.com/main). I had just served 2 years in the Army and also had just come back from the [ISC](http://isc-hpc.com) that was held in Germany for a week.

My flight landed the day before the conference so I was sleep deprived and so tired that I couldn't really say all the things I had in my head. I was not prepared for this and maybe I was almost arrogant to think that I would be able to pull this off.

Oh well, at least I learned something from this. At least I won't make the same mistake for my next talk! 🤦‍♂️

