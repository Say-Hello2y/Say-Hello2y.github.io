---
layout: "post"
title: "Contributing to Swift (한글 버젼)"
date: "2017-08-10 09:32"
description: "A how-to guide"
comments: true
hide: 1
tags:
- swift
- open-source
---

스위프트를 계속 사용하다 보면 스위프트가 오픈소스라는 사실이 종종 생각나곤 한다. 그리고 스위프트를 더 계속 사용하다 보면 스스로가 이 오픈소스 프로젝트에  어느 정도라도 기여하고 싶다는 순수한 마음이 생길 수도 있다. 이런 애정어린 마음을 갖고 애플의 [스위프트 레포](https://github.com/apple/swift)를 찾아 가지만 대부분의 사람들은 이내 어마어마한 코드 베이스에  위축되어 그냥 포기하곤 한다.

당연히 사용자의 한 사람으로서 스위프트라는 오픈소스 프로젝트에 기여하는 것은 진정 보람있는 것이라 할 수 있다. 다만 이러한 작업이 쉽지 않다는 것만은 분명한 사실이다. 본 블로그 포스트에서는 스위프트 오픈소스에 관심있는 사람이 어떻게 하면 스위프트에 기여할 수 있는지에 대해서 살펴보고자 한다.

# 개요
가장 먼저 스위프트에 기여할 수 있는 방법을 5가지의 작은 부분으로 구분하였다.

1. 버그 (문제점) 찾기
2. 프로젝트 빌드 하기
3. 코드 짜기
4. 테스팅 하기
5. Pull Request 올리기

이번 포스트에서는 위 1번과 2번에 대해 언급하려고 한다.

# 스텝 1: 버그(문제점) 찾기
필자는 Xcode에서 작업을 하다가 찾은 auto fixit 버그 때문에 스위프트에 기여하기로 결심을 하였다.   필요한 곳에 `@escaping`을 자동으로 넣어주는 fixit이 작동을 하지 않는다는 사실을 알게되었기 때문이다. 본 버그를 확인한 후 문제점을 식별하기 위해 계속 실험을 해보니 스위프트 컴파일러는 함수 파라미터에만 fixit 행동을 제공하고 있었고, 반면 필자의 코드는 함수에서 리턴을 하는 클로져의 파라미터에 에러가 발생했기 때문에 자동 fixit이 제공되지 않았던 것이다. 

```swift
func mapping <A, B, C> (f: @escaping (A) -> (B)) -> (@escaping ((C, B) -> (C))) -> (C, A) -> (C) {
    return { reducer in
        return { accum, input in
            reducer(accum, f(input))
        }
    }
}
```

리턴되는 `((C, B) -> (C)) -> ((C, A) -> (C))`에서 `@escaping`을 빼면 오류를  재현할 수 있다.

> 설사 코드를 이해하지 못한다 해도 중요한 것은 스위프트에 어떻게 기여하는가를 배우는 것이 중요하기 때문이다. 그래도 궁금하다면 **[Transducers in Swift]({{ site.url }}{% post_url 2017-07-17-transducers %})** 포스트를 참고하기 바란다.

### 스위프트 커뮤니티 물어보기
문제는 찾았는데 그 다음 해결단계가 뭔지 몰랐다. 그래서 주변 다른 사람들에게 물어 보기로 결심하고 `swift-dev@swift.org`에 질문을 던졌다. 그리고 답장이 왔다.

```
This has been reported as SR-5556.
In the future, please report bugs like this through bugs.swift.org and/or Radar.

Thanks,

~ Robert Widmann
```

버그 발생시에는 이메일을 하지 말고 `bugs.swift.org` 또는 Radar에 리포트를 해야 된다는 것이었다.

*리포트는 영어 이메일로 작성해야 되지만 영어를 못하는 외국인들도 많기 때문에 대체적으로 잘 받아 주는 경향이 있다.*

## OK사인을 받아라
버그 리포트를 작성하고 나서는 내가 할 수 있는 것은 다했다고 생각했다. 그런데 2주가 지난 후 버그에 대한 상황 파악을 하려고 사이트에 들어가 봤는데 아무런 조치도 없는 상태였다. 순간 나는 ‘애플 직원들은 큰 기능들을 프로그램하기 바빠서 이런 작은 버그들은 신경도 안 쓰는가 보다’ 라는 생각을 했다. 

이유야 어떻든 내가 찾은 버그이니 내가 고치겠다는 마음을 먹고 버그를 담당하고 있는 사람에게 질문을 던졌다. 그리고 내가 직접 버그를 수정해도 된다는 "Ok!"를 받았다. 

{: .center}
![JIRA comment]({{ site.url }}/assets/swift_contrib/jira_comment.png)

이때 순간 나는 "아  망했다. 왜 괜히 물어 봤지?" 라는 생각을 했다.

# 스텝 2: 프로젝트 빌드하기
이 부분은 그냥 잘 따라 하기만 하면 된다. 그런데 프로젝트 빌드 스텝이 계속 바뀔 수도 있으니 반드시 [깃험 스위프트 레포](https://github.com/apple/swift/blob/master/README.md)를 참고하기 바란다.

리드미도 좋지만 리드미에 상세 설명이 되어있지 않은 부분들은 이해가 쉽도록 추가해서 정리 해봤다.

> 아래 스텝들은 맥을 사용하는 사람에게만 해당이 되기 때문에  만약 리눅스를 사용하는 사용자가 있다면 리드미를 참고하기 바란다. 그렇지 않으면 맥을 사는 것도...


1. `Homebrew` 설치
  - `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
2. `brew install cmake ninja`
  - 위 패키지들은 나중에 스위프트를 빌드할 때 사용된다
3. `mkdir swift-source & cd swift-source`
    - 이 스텝을 안하면 다음 스텝에서 디렉토리 구조를 엉망이  될 수 있으니 **꼭** 하도록!
4. `git clone https://github.com/apple/swift.git`
5. `./swift/utils/update-checkout --clone`
   - 스위프트를 빌드하기 위한 다른 레포들을 클론하는 스크립트
    - 오래 걸리니 좋은 책을 하나 선택해서 읽어보도록!
6. `sudo xcode-select -switch /Applications/Xcode_ver_num.app`
   - **주의** 위 스텝을 실행하기 전에 [system requirements](https://github.com/apple/swift/blob/master/README.md#system-requirements)을 **반드시** 읽고 올바른 Xcode 버전를 다운로드 하고 설치 하도록!
7. `./swift-src/swift/utils/build-script -x`
    - 스위프트를 빌드 하고 작업할 수 있는 Xcode 프로젝트 생성

{: .center}

![CAUTION! Hot!]({{ site.url }}/assets/swift_contrib/cpu.png)

마지막 빌드하는 스텝동안은 컴퓨터가 몇 시간 동안 매우 뜨거워 질 테니 사전에 골라놓은 책을 읽으며 여유있는 시간을 보낼 것을 추천한다. 빌드가 다 끝나면 `./swift-source/build/Xcode-DebugAssert/swift-macosx-x86_64` 에 `Swift.xcodeproj` 라는 파일이 있는지 확이해라.

![Umm...]({{ site.url }}/assets/swift_contrib/xcode.png)

이제 부터 우리가 흔히 사용해 왔던 Xcode에서 작업을 하게 될 겄이다. 그러나 Objective-C 도 Swift 도 아닌 C++ 로… 😱

# 스텝 3: 코드 짜기 (To be continued...)
자 이제 코드를 짤 시간이 되었다. 이 부분은 사람마다 다르겠지만 미약하나마 이 글을 보는 분들에게 도움이 될 것을 기대하며 필자의 경험을 👉👉[**다음 블로그 포스트**]({{ site.url }}{% post_url 2017-08-24-contributing-to-swift-2 %})에  담아 봤다.

## ⚠️ 참고

위에서 설명한 버그를 자세히 본 애플 직원이 이 버그를 고치기 쉽지 않다는 판단을 내렸고 나에게 조언을 해주었다. 그래서 나는 조금 허무하게도 다른 버그(https://bugs.swift.org/browse/SR-910)를 고치는 것으로 방향을 급 수정했다. 여기서 우리가 배울 수 있는 또 다른 중요한 것은 

> 어려운 부분에서 막히면 고민만 하지 말고 주변의 다른 사람들에게 물어봐야 한다는 것이다.
