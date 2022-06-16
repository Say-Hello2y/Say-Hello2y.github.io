---
layout: post
title: "Reentrant and Threadsafe Code"
date: 2017-09-17 13:28:03+0900
description: "Through the gates of multithreaded hell"
comments: true
tags:
- c
- multithreaded
---


If you have programmed in C, you probably have typed `man strtok` on the terminal. Ah yes, the infmaous `strtok` function. The internet seems to really hate it but you decide to give the function a chance and try to read through the man page. Just in case none of what I said applies to you, here's the first sentence in the description section of the man page.

> **This interface is obsoleted by** `strsep(3)`

{: .center}

![Well then...](https://media.giphy.com/media/3o7TKQ8kAP0f9X5PoY/giphy.gif)

Uh ok... Even the folks at FreeBSD seem to hate it so much that they just completely replaced it with a different function. But you keep reading to see what's so bad about it and you come across the following sentence.

> The `strtok_r()` function is a reentrant version of `strtok()`

At this point, you could asking *"what does it mean by a function to be reentrant?"*. If you know the answer to this question, you should stop reading this post and move on to a [different]({{ site.url }}{% post_url 2017-08-10-contributing-to-swift %}) post.

For those who don't know what reentrant functions are, this is how the man page describes it.

> The context pointer last must be provided on each call.  The `strtok_r()` function may also be used to nest two parsing loops within one another, as long as separate context pointers are used.

Does it mean that the reentrant function is some how better than the non-reentrant version because it can be used to nest two parsing loops? Well, before we can answer that question, we first need to know what makes a function reentrant than its non-reentrant counterparts.

# Threadsafe Code

From the word *"reentrant"*, you can kind of guess that it is probably going to be some piece of code that is going to be *re-entered*. And you would be right. Reentrant code is mostly used in multithreaded programs to maintain its integrity and keep programmers from going insane from mysterious race conditions and pure hell. And now, you could asking *"Hey! Don't we already have a word for that? That kind of code is called threadsafe!"* And , you would be right, once again. **So what's the fricking difference???**

## Reentrant Code VS. Threadsafe Code

Let's do some definitions.

> Thread safe code is one that can be **performed from multiple threads safely**, even if the calls happen **simultaneously** on multiple threads.

> Reentrant code is one that **can be entered by another actor before an earlier invocation has finished**, without affecting the path the first action would have taken through the code.

Did you catch the difference? Thread safe code means you can call the function on multiple threads. Reentrant code means that you can do all the things thread safe code can do but also gurantee safety **even if you call the same function within the same thread.** 

So, reentrant code can be thread safe but thread safe code can't be reentrant? Not necessarily… Before we complicate things, let's go and look at some code.

# Show me the code

Just like how most books do it, we will first start off with bad code. Then, we will slowly make it better, worse again, and eventually make it immune to any convoluted multithreaded code.[^1]

## Reentrant ❌ | Thread-safe ❌

```c
int t;

void swap(int *x, int *y) {
  t = *x;
  *x = *y;
  // `my_func()` could be called here
  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

- ❌  Thread-safe because
  - Global variable `t` is constantly mutating within `swap`
- ❌  Reentrant because
  - Global varaible `t`
  - `my_func()` could be called while `swap()` is running in the same context. If so, value of `t`  would be unpredictable

This code can easily be made to be thread-safe though. All we need to do is make `t` a thread local variable.

## Reentrant ❌ | Thread-safe ✅

```c++
#include <threads.h>
// `t` is now local to each thread
thread_local int t;

void swap(int *x, int *y) {
  t = *x;
  *x = *y;
  // `my_func()` could be called here
  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

-  ✅  Thread-safe because
  - Variable `t` is local to each thread
-  ❌  Reentrant because
  - Threads are safe from each other but if multiple calls to `my_func()` happen within a single thread, value of `t` is still unpredictable

##  Reentrant ✅ | Thread-safe ❌

No one would do this in real life but for the sake of example, here's some very convoluted code.

```c
int t;

void swap(int *x, int *y) {
  int s;
  // save global variable
  s = t;
  t = *x;
  *x = *y;

  // `my_func()` could be called here
  *y = t;
  // restore global variable
  t = s;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

- ❌ Thread-safe because
  - Global data can't be guaranteed to be the same at all times
- ✅ Reentrant because
  - Funky but global data is the same when the program enters and or leaves `swap`

##  Reentrant ✅ | Thread-safe ✅

The solution was just staring at us from the beginning. That one professor from CPSC 101 was right all along; global variables are bad.

```c
void swap(int *x, int *y) {
  int t = *x;
  *x = *y;

  // `my_func()` could be called here
  *y = t;
}

void my_func() {
  int x = 1, y = 2;
  swap(&x, &y);
}
```

-  ✅ Thread-safe and  ✅ reentrant because
  - `t` is now allocated on the stack

# Rules of Thumb

To write a block of code that is reentrant, you should adhere to the following rules of thumb[^1]

1. Don't use global or `static` variables
2. Don't let it modify its own code
3. Don't call other non-entrant functions

# Back to strtok

So, let's apply what we learned here to `strtok`. Usually, when we use `strtok`, we do the following[^2]

```c
token[0] = strtok(string, separators);
i = 0;
do {
  i++;
  token[i] = strtok(NULL, separators);
} while (token[i] != NULL);
```

This is completely reasonable code besides the fact that it is using `strtok` and is neither reentrant nor thread safe. However, `strtok_r` can come in and save the day with only minor changes to the code

```c
char *pointer;
...
token[0] = strtok_r(string, separators, &pointer);
i = 0;
do {
  i++;
  token[i] = strtok_r(NULL, separators, &pointer);
} while (token[i] != NULL);
```

Now, thanks to the reentrant version of the `strtok`, this code block can now be used in multithreaded programs.

# So...

the next time you are stuck in constant deadlock, data mutating, blood boiling mutiprogramming hell, first ask yourself, *"Hey, is this code reentrant and or threadsafe?"*

[^1]: [Wikipedia](https://en.wikipedia.org/wiki/Reentrancy_(computing))

[^2]: [IBM Knowledge Center](https://www.ibm.com/support/knowledgecenter/en/ssw_aix_61/com.ibm.aix.genprogc/writing_reentrant_thread_safe_code.htm#writing_reentrant_thread_safe_code__mfr)
