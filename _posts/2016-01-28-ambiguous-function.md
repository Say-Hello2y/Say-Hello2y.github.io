---
layout: post
title:  "The Ambiguous Function"
date:   2016-01-28 11:49:23 +0900
description: "The angelic operator"
comments: true
tags: 
- AI 
- lisp
---

John McCarthy is known for being the pioneer of Artificial Intelligence. His work in Stanford and his creation of Lisp during the 50’s is something that was far ahead of the time. Here's a fun fact. Lisp, the ‘language’ John McCarthy created, was not supposed to be a programming language but something he created for himself to describe algorithms and functions in a ‘mechanical’, and a ‘logical’ way. Only after his death did graduate students at MIT create it into a computer programming language.

So, with a language so deeply rooted in Artificial Intelligence, here’s another mind blowing fact. Have you ever heard of a function called `(amb)`?

## Definition
According to the “SchemeWiki”, **amb** is the ‘ambiguous’ special form for non-deterministic computationl; syntax is `(amb expression…)`

It evaluates and returns the value of any expression operand. As code continues, the set of values that it may have returned is typically narrowed with a (REQUIRE condition) procedure that ensures that condition holds true. That was rought. Let's try that in plain English.

## Less wikipedia-ish definition
Amb is non-deterministic, which means the result of the function call has not been determined yet before the function call. So even with the same input, the output of the function depends on the situation and the state of the program. 

But what influences the function’s decision to either return one result or another? Now, this is where things get interesting.

> Amb’s purpose is not just to return something ‘random’ but to return a value that would lead the entire function to succeed.

Hence, its nickname as the 'angelic operator'. 
That’s pretty rad. Let’s look at some examples.

## Examples
To start off with, here's a basic every-day implementation of amb.

```scheme
(amb #t #f)
```

Now, this line may return either true or false, as once again, amb is a non-deterministic function. But in this case, the expression would return true. The function is provided with a `#t` true and a `#f` false.  as the first value it sees is true and by returning it, the overall program would be allowed to exit successfully.

Now, one must know that the call

```scheme
(amb)
```
with no expressions has no value to return, and is considered to fail.

So the following calls,

```scheme
(amb 1 (amb))
(amb (amb) 1)
```

would also fail miserably.

Now, check this out. Remember the thing about amb being an angelic operator and how one of its purposes was to make the program as a whole succeed?

```scheme
(if (amb #t #f)
    1                   ;executes if true
    (amb))              ;the else block (executes if false)
```

We already talked about the function inside the if-statement `(amb #t #f)` and how it could return either true or false. However, if amb returns a false in this case, (amb) would execute, leading the entire program to FAIL.

So, in this case, amb has to always return a true in order for the value 1 to be returned, allowing the entire program to succeed. Awesome, right?

Now you may be wondering, that sounds cool and all but what can I do with this?

## Applications

First of all, lets start off with something easy and familiar. You guys may have heard of the function “assert”, commonly used in C or C++ for checking if something is NULL or not.

In Scheme, you can use a macro function in order to come up with our own version of the assert function.

```scheme
(define our-assert
  (lambda (statement)
    (if (not statement) (amb))))
```

Essentially, this function is saying that `(our-assert statement)` has to be true and if not, it will cause the program to fail by calling the function (amb) with no arguments.

Ok, cool. That was exciting but what else can we do with this? I talked a little about A.I’s in the beginning and so does that have anything to do with it? Ummm, kinda. Have you guys heard of the Kalotan Puzzle or the concept of map coloring? You can check them out here but basically, it allows us to write code that is almost impossible (or very hard) to write in C-based languages.
