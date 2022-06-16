# Magic behind Ctrl-C, Ctrl-V

If you are reading this article, you probably have used `Ctrl-C` and `Ctrl-V` in order to copy string into a text document. Whether if it was trying to plagiarize that one mid-term report you really didn't feel like doing or copying some code from StackOverflow, we have all done it numerous times. But have you ever wondered how the copy and paste feature actually works? And have you ever thought about how you can undo and redo everything you have ever typed like a time machine?

Tthe answer to this question is a data structure called piece table. Piece table was invented by J. Strother Moore, a computer scientist who worked at Xerox PARC in 1974,  as a side-consequence of work he did on representing logic clauses within software that proved theorems. I tried to find more information regarding the matter but the interwebs seems like it's not too interested in it. There is a cool article here that talks a little about the piece table's background and how it can be implemented, but that's just about it. 

# Implementations

To see the benefits of piece tables, let's try to build the copy and paste feature of our imaginary text editor. But before jumping straight into the piece table implementation, let's start with a naive implementation of the feature.

## Naive Implementation

The approach here is simple; represent the entire text as one long string of characters. Then, if we want to insert a string at index 133, we can go to that index, make some room, move all characters to the right by the number of characters to be added, and then copy the new string into the newly created gap.

In sudo C code,

```c
char *longStringBuf = "lorem ipsum blah blah something interesting and long here";
char *insertPoint = longString + insertIdx;

// Make room for new string
char *expandedBuf = realloc(longStringBuf, strlen(longStringBuf) + newStrLen);
// Move existing string to make gap for new string
memmove(insertPoint, insertPoint + newStrLen, newStrLen);
// Copy string into gap
strcpy(insertPoint, newStr, newStrLen);
```

Imagine running various versions of the above code every time you want to paste or undo-paste. Ugh….

## Idea behind Piece Table

When you read algorithm books, they always talk about the "divide and conquer approach". It's the concept they use to teach people about merge sort and binary search and turns out, it applies here as well! The idea is simple;

> Instead of having a single gigantic string, why not just have bunch of small set of structs/record that consist of only a few bytes of data that describe how a text string is fragmented.

## How would it work?

### 1. The struct

What kind of data would this struct contain? First, we can start off by making a variable that holds the characters that belong to a single record. 

### 1. Empty Slate

When we first paste a string into an initially empty document, we would create a record with the format we just talked about above. We will call this format a piece description. 

https://web.archive.org/web/20160308183811/http://1017.songtrellisopml.com/whatsbeenwroughtusingpiecetables

