---
layout: post
title: "Contributing to Swift II (한글 버젼)"
date: 2017-08-24 13:10:42+0900
description: "github.com/apple/swift"
comments: true
hide: 1
tags:
- swift
- open-source
---

1. ✅ 버그 (문제점) 찾기
2. ✅ 프로젝트 빌드 하기
3. 코드 짜기
4. 테스팅 하기
5. Pull Request 올리기

1번과 2번은 [이전 블로그 포스트에]({{ site.url }}{% post_url 2017-08-10-contributing-to-swift %}) 설명이 되어있다. 만약 읽지 않았으면 이 글을 보기 전에 먼저 읽어보기를 권한다.

# 스텝 3: 코드 짜기

이전 포스트의 마지막 부분을 보면 필자가 애초에 고치려고 했던 버그 대신 다른 버그를 고쳤다는 것을 알 수 있다. 새로운 버그에 대해서 설명 해보겠다.

새로운 버그는 설명하기가 조금 더 쉽다. 존재하지 않는 변수를 tertiary expression에 제공하는 경우 컴파일러는 예상외의 이상한 오류 메세지를 보낸다.

```swift
enum E {
    case LessThan
    case EqualTo
    case GreaterThan
}

func compare(integer lhs: Int, with rhs: Int) -> E {
    guard lhs != rhs else { return .EqualTo }
    return lhs < rhs ? .LessThan : .GreaterTham
}

/// Original compiler diagnostics
>> error: cannot convert return expression of type 'Bool' to return type 'E'
    return lhs < rhs ? .LessThan : .GreaterTham
           ~~~~^~~~~

/// What we want
>>  error: type 'E' has no member 'GreaterTham'
    return lhs < rhs ? .LessThan : .GreaterTham
    note: did you mean 'GreaterThan'?
```

자, 이제 뭐를 어디서 부터 어떻게 고쳐야 되는 것일까?

일단 가지고 있는 스위프트 컴파일러가 우리가 예상하고 있는 틀린 에러를 표시하는지 확인해야 된다. 그런데 Xcode에 무슨 수로 `.swift` 파일을 제공하고 그 파일을 컴파일 시킬 수 있을까?

## Xcode안에서 스위프트 컴파일 하기

1. Xcode scheme 중에서 `swift` 선택
2. `Scheme > Edit Scheme`
3. `Run -> Arguments -> Arguments Passed On Launch -> -frontend -typecheck /PATH/TO/.swift`

![Scheme customization]({{ site.url }}/assets/swift_contrib/edit_scheme.png)

이제 빌드를 `⌘ + b` 하면 Xcode debug console에 컴파일 결과가 출력된다. 

## 버그 고치기

실제로 버그를 고치는 방법은 버그 마다 다를 수 밖에 없다. 그래서 여러분들이 버그를 고치는 작업에 도움이 되었으면 하는 바램으로 필자의 경험담을 얘기해보겠다.

### a. 어디서 부터 시작을 해야 될까?

처음에는 이 고민 부터 했다. 몇 천개가 넘는 C++ 파일들중에 버그가 있다고 상상하니 찾아나가는 과정에 희망이 없어 보였다. 그래서 무작정 `grep`을 사용하여 산더미처럼 쌓인 파일들을 검토해 나가기 시작했다.

마침내 컴파일러가 출력하는 이상한 메세지를 다음과 같은 명령어로 검색했다.

```bash
cd swift-src/swift & grep -R 'cannot be resolved without a contextual type' * | less
```

그 중에 눈에 띄는 파일이 있었다. `include/swift/AST/DiagnosticsSema.def` . 이 파일이 특별해 보였던 이유는 크게 없다. 다만 `diagnostics`라는 단어가 들어가 있었기 때문에 버그와 관련된 파일일 수도 있다는 느낌이 들었다.

### b. 파일 찾기

`DiagnosticsSema.def` 을 보면 실제로 컴파일러에서 사용되는 에러, 경고, 노트 문구들 밖에 없다. 컴파일러가 내용을 출력하기 전에 마지막으로 거치는 파일이기는 하지만 실제로 버그가 있는 것은 아니다. 그래서 터미널을 다시 켜서 `grep`을 타이핑하기 시작했다. 다양한 키 워드를 넣어서 수백번 넘게 검색한 것 같다.

몇 시간 후, `FailureDiagnosis`라는 클래스를 발견했다. 이름도 그럴 듯 하고 코드도 왠지 문제와 관련이 있어 보였다. 그런데 이것은 모두 내 초보자의 생각에서 나오는 추측일 뿐. 전문가의 의견이 필요했다. 그래서 나는 미국 사람들이 많이 사용하는 트위터를 통해 애플 스위프트 컴파일러 팀에 있는 사람에게 트위트를 보냈다. 그리고 몇 분도 안돼서 답변이 왔다

> ㅇㅇㅇ (한글 번역)

**다시 한번 더 강조하고 싶다. 질문이 있으면 관련있는 사람들한테 물어 봐라. 트위터가 없으면 트워터에 가입해서 사람들한테 Direct Message (DM)을 보내고 그게 싫으면 `swift-dev@swift.org`에 이메일을 통해 질문을 던져라.**

{: .center}
![톰 아저씨도 도와주고 싶어 한다](https://media.giphy.com/media/uRb2p09vY8lEs/giphy.gif)

### c. 버그 고치기

드디어 어려운 과정을 거쳐 필자는 버그를 고쳤다. 그러나 고치는 과정이 그리 수월하지만은 않았다.  [@**CodaFi_**](https://twitter.com/CodaFi_)라고 하시는 분이 있는데 이 분은 스위프트 컴파일러 팀에서 일하시는 분이다. 그분의 트위터  페이지를 보면 대단한 사람이라는 것을 곧 바로 알아낼 수 있다. 필자는 이 사람한테 질문을 던졌고 신속하고 정확하게 답변을 받았다.

![The Answer]({{ site.url }}/assets/swift_contrib/convo.png)

# 스텝 4: 테스팅 하기

버그를 고치면 바로 pull request (PR) 을 올리고 싶겠지만 참아야 된다. 일단 [본 항목](https://github.com/apple/swift/blob/master/docs/Testing.md) 부터 읽어야 한다. 이 항목은 애플에서 제공하는 "테스팅하는 방법"이며, 자세히 보면 매우 유익한 정보가 많지만 지면 관계상 필자가 내용을 간단히 요약해 봤다.

1.   `utils/build-script --test`

   - 위 스크립트는 **모든** 테스트를 실행한다. 또한 그래서 시간이 매우 오래 걸린다 (2~3 시간). 처음부터 컴파일러 빌드를 다시 해야 하기 때문이다
   - 코드를 많이 작성하거나 또는 고쳤다면  `—validation-test`와 같이 실행하기를 바란다
   - 빌드는 바뀐 코드들만 부분적으로 컴파일 하지만 안 그럴 경우 `ninja swift`와 `ninja swift-stdlib`로 직접 수동 컴파일을 해도 된다

2. `lit.py` testing

   - 위 파이썬 스크립트는 원하는 파일을 선정해서 그 파일만 테스팅하게 해주는데, 특정한 파일들만 테스팅하기 때문에 결과가 빨리 나온다. *개발하는 동안은 이 방법을 사용하는 것을 추천한다*
   - 워낙 많은 옵션들을 제공하기 때문에 `lit.py -h`을 참고하면 된다
   - 자주 사용하는 방법은 아래와 같다

   ```shell
   # 컴파일러 수동 빌드
   cd swift-source/build/Ninja-DebugAssert/swift-macosx-x86_64 & ninja swift

   # diagnostics.swift 파일만 테스트
   # 현재 디렉토리는 lit 테스팅을 할 때 매우 중요하다!
   llvm/utils/lit/lit.py -sv ./build/Ninja-ReleaseAssert/swift-macosx-x86_64/test-macosx-x86_64 --filter=diagnostics.swift
   ```

3. `cmake`

가능하지만 리드미는 다음과 같은 경고를 한다.

```
Although it is not recommended for day-to-day contributions,
it is also technically possible to execute the tests directly via CMake. 
```

따라서 만약 당신이 초고수가 아니라면 1번과 2번을 사용하는 것이 좋을 것 같다.

> 테스트를 작성 할 때는 코드를 짧게 짜는 것을 추천한다. 테스트 하려고 하는 기능과 관련 없는 불필요한 코드는 가능하면 없는 것이 좋다

테스트가 "올 패스" 하기를 기다리면서 PR을 올릴 준비를 하자.

![올 패스~](http://cfile26.uf.tistory.com/image/2549E549573F9C8A2909C3)

# 스텝 5: Pull Request 올리기

이 단계는 코드를 예쁘게 정리하고 애플에서 검토하는 사람들이 잘 검토하게 하기 위해  다듬는 단계다. 먼저 애플에서 제공하는 [가이드라인을](https://swift.org/contributing/#contributing-code) 먼저 참고하기 바란다. 유용한 정보도 많고 해야 되는 것들도 많다. 

그래서 필자는 PR을 올리기 전에 살펴 볼 수 있는 체크리스트를 만들어 봤다.

1. 커밋
   - 작성하거나 바꾼 코드가 한 파일에 집중되어 있다면 커밋 타이틀에 `[tag]`을 포함하라. 예를 들어 `[stdlib] ......` 
   - 커밋 메세지에는 `Resolves: SR-xxxx`을 포함
   - 모든 커밋을 한개의 커밋으로 "squash"; `git rebase` 참고
2. Indentation
   - 탭 대신 스페이스
   - 탭은 스페이스 2개
3. 포맷
   - 필요 없는 whitespace 제거
   - 커밋하기 전에 `clang/utils`에 있는 `git-clang-format` 실행

# 끝

마지막으로, 필자의 [첫 PR을](https://github.com/apple/swift/pull/11531#pullrequestreview-58832254) 올려봅니다. 화려하지는 않지만 나에게는 꽤 큰 성취감을 가져다 준 PR이었습니다.