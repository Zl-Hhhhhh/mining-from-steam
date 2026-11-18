# Topic seed units and boundary review

**候选**：`sentence_k10_s42`  
**状态**：已采用建议版本，等待人工校准  
**用途**：确认每个主题的语义原型和边界，再进行句子级多主题标注。

## 审查方式

每个主主题先提供 5 条真实句子作为原型，覆盖中英文、正负表达和不同证据类型。边界样本单独列出，用来确认多主题、泛评价和技术问题的处理方式。

请对每条种子填写以下判断。

- 保留、替换或拒绝。
- 批准的主题 ID。
- 是否适合作为该主题的原型。
- 需要补充的边界说明。

这些种子只用于形成主题原型，不会直接把相同标签扩展到所有句子。

## 主主题种子

### T01 玩法与机制

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S01b9954a79a22eb0a857` | `The formula Slay the Spire has with its well-executed roguelike mechanic combined with satisfying deckbuilding and combat makes it such a good time-waster` | 机制、构筑和战斗循环 |
| `S0133c4c7b8c8e5f21cc7` | `I enjoy that you can just chill and go mine for resources` | 资源采集行为 |
| `Sb42503c567c1948611df` | `Lot of skill and build options too` | 技能和构筑选择 |
| `S0c569b921f7e91bbb5b0` | `The play at your own pace, while you explore, catch, build, craft, grind, and level up gives you tons of options.` | 探索、制作和升级，也涉及 T08 边界 |
| `S00c82cc5bf031552f720` | `战斗系统真是一坨，重制了再次赤石` | 负面战斗系统 |

建议边界是“具体描述怎么玩”才进入 T01。只有“好玩”“上瘾”时走泛评价路由。

### T02 操作、难度与平衡

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S002c87c6de9fcc8c1b3e` | `Difficulty: Absurdly punishing` | 难度和惩罚 |
| `S00ba2db7404b21f27522` | `新手教程很完整` | 教程和上手 |
| `S01c16c376643900659d7` | `You must learn the boss mechanics by heart, and the game doesn't teach you those mechanics. * It is extremely imbalanced` | 学习门槛、教程和平衡的多主题边界 |
| `S00a1ed87813f3ac7236e` | `极致的打击感` | 操作手感 |
| `Sa0e5a996ba89bc396370` | `The game is difficult` | 短句难度证据 |

需要确认“打击感”是否始终归 T02，以及难度与玩法机制同时出现时是否保留双标签。

### T03 性能、稳定性与故障

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S0182e47889494f60224a` | `i cant connect to ea servers` | 服务器连接 |
| `S034ac1199998ca0515cc` | `Cant fix lost connection bug for over a year` | 断线故障 |
| `Sd7cfc20e1ae65f34d393` | `老是闪退，垃圾游戏` | 闪退 |
| `Sb242d02bad1271517d34` | `重置版后出现各种各样的BUG，重要还要经常卡死，玩的真TM闹心` | BUG 和卡死，也涉及 T08 边界 |
| `S0772d56f4a64811e0436` | `在主界面崩溃，闪退` | 启动崩溃 |

需要确认服务器连接和联机体验同时出现时是否双标签。更新造成的故障应归 T03，更新内容和维护支持归 T08。

### T04 剧情与角色

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S005f3d894306bfef1e5e` | `And THIS Game is still amazing, it still has all the depth of story telling, characters that we can love and hate and form our own truths with` | 叙事和角色 |
| `S00cea81f3bae5ac28808` | `It's very underrated because the main story takes a lot to gain momentum and be any good writing-wise,` | 主线写作 |
| `S00e03e01e9c1bb85c717` | `剧情方面确实不太好，` | 短句剧情证据 |
| `S07d37d6c2b23c1b80d10` | `Geralt is one of the best protagonists in gaming, and his relationships with Ciri, Yennefer and the other characters make rhe game even better` | 主角和角色关系 |
| `Sfb6ca2c44526b75ed437` | `The story line is fun to follow as well` | 叙事中心代表句 |

需要确认世界观、对白和角色关系是否统一归 T04。只有台词或角色名而没有评价语义时不强行归类。

### T05 画面、美术、音乐与音效

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S001477cbee3a3273aee8` | `With music it'll make you tear up with how beautiful it is` | 音乐 |
| `S00d81e425cd36be8097c` | `The art style is fantastic and so immersive` | 美术风格 |
| `Sfb2228f0feda41d3cc55` | `The games graphics also just not good at all` | 负面画面 |
| `S3fcd7609b6e1db71abfd` | `nice game the visuals and environments are great` | 视觉和环境 |

需要确认“沉浸感”只有在有视听证据时才归 T05。画面、音乐和动画是否保留为一个主题也需要审查。

### T06 联机、匹配、作弊与社区

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `Sd998edafeabe8e855ae6` | `和朋友一起玩很好玩` | 朋友合作 |
| `S0173350e5b0e9b024360` | `its more fun with friends and also dont use discord or any other vc, use the ingame vc its good and thats how its meant to be played` | 合作和语音 |
| `S0ca1ba41050c18513c02` | `Cheating is absolutely atrocious and yet we are told relentlessly that it is being, "looked into." Horrible devs ruined what could have been a wonderful game. 0/10` | 作弊和治理 |
| `S00829350bcf162a9f5c0` | `A perfectly executed team fight, a game-winning Black Hole, an unexpected Rampage, or that one teammate who actually knows what they're doing` | 团队对战和队友 |
| `S01921a60e08137e660b3` | `Too toxic to enjoy.` | 社区环境，低权重 |

需要确认没有联机语境的辱骂是否全部走 A01 或 A03。延迟和服务器故障可以同时进入 T03。

### T07 价格、付费与 DLC 收费

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S00dfec322186d9328513` | `大表哥好像史低七十多，小黑盒和ST对照着看，说不定GTA6出来了会更便宜` | 历史低价和购买时机 |
| `S032c9c28326b6dcd1dd7` | `I bought this game again after having it on ps4 for a while, worth every dollar` | 购买价值 |
| `S0390eb3cf0fc20303be8` | `I shall spend some $ on them microtransactions, i want to support this otherwise 'free to play' game` | 内购 |
| `S07474510ae7753b543cd` | `If you like souls-inspired boss rushes, I'd suggest buying at a fairly discounted price.` | 折扣购买 |
| `S075ed5f3567901a274df` | `It is a deeply disappointing entry that completely abandons key series staples, offering a thin, unpolished package that fails to justify its full asking price` | 价格与内容价值，也涉及 T08 |

需要确认“DLC 内容太少”和“DLC 太贵”是否分别进入 T08、T07，二者同时出现时保留双标签。

### T08 内容量、重玩、MOD 与更新维护

| unit_id | 句子 | 建议用途 |
| --- | --- | --- |
| `S00e7e93b06e8b017a492` | `如果能到2077年还在更新就好了，会是本游戏最酷最赛博的一点` | 长期更新 |
| `S0189cee3fc86e0432321` | `横向内容很多，` | 内容量，低权重 |
| `S01bb7450909d0ecc1798` | `this game is cooked without 3rd party mods` | MOD 依赖 |
| `S0734dbe02611a1f88291` | `The dev supports mods for the game which is always a plus because it allows personalization in your experience` | MOD 支持 |

需要确认“玩了很久”“上头”不能单独证明重玩性。更新后闪退仍应进入 T03。

## 边界样本

| unit_id | 句子 | 建议路由 | 需要确认的边界 |
| --- | --- | --- | --- |
| `S4be3cb14ee5f4d2ae51c` | `好玩，很有成就感` | `A01` | 没有具体方面证据的泛评价 |
| `S8d82aebf3dabefcddc68` | `Graphics are amazing and the gameplay is too` | `A02`，`T01`+`T05` | 玩法和视听双标签 |
| `S73108312f5201eef4f9f` | `This game's story and gameplay are both so enjoyable` | `A02`，`T01`+`T04` | 剧情和玩法双标签 |
| `Sd6a86b001ac0e9c0c6b6` | `更新后直接玩不了了` | `A02`，`T03`+`T08` | 更新支持和更新故障的区分 |
| `S72f9870a8c1745977b93` | `买这个游戏到现在，玩都没有50局，游戏突然崩溃已经20局了，游戏闪退之后重新登陆，还要给我禁赛，你们他妈的能不能把服务器优化好` | `A02`，`T03`+`T06` | 购买只是背景，具体方面是崩溃、服务器和禁赛 |
| `S305423fb44d7c33081db` | `the game is to slow, to expensive and the story isn't enjoyable` | `A02`，`T04`+`T07` | 价格和剧情明确，slow 的含义暂不强行归入 T02 |
| `S01c16c376643900659d7` | `You must learn the boss mechanics by heart, and the game doesn't teach you those mechanics. * It is extremely imbalanced` | `A02`，`T01`+`T02` | 机制、教程和难度平衡共现 |
| `S081b17aeb61cfe04da40` | `I love the animation and voiceovers, the story is amazing and super entertaining` | `A02`，`T04`+`T05` | 剧情和视听共现 |
| `S0889e8a7b511f0c66fa5` | `反作弊报错` | `A02`，`T03`+`T06` | 技术报错和反作弊系统共现 |
| `S027c0d0036fa3b2bd43c` | `There is never a death i have disagreed with and the soundtrack beautiful` | `A02`，`T01`+`T05` | 游戏判定或机制与 soundtrack 共现 |
| `S000d0b420c15930573d2` | `I would happily pay for more expansions of this quality` | `A02`，`T07`+`T08` | 扩展内容和付费意愿共现 |

## 你审查后需要决定的事项

1. 每个主题保留哪些种子，替换哪些种子。
2. T05 是否继续把画面、美术、音乐和音效合并。
3. T07 与 T08 遇到 DLC 时的双标签边界。
4. T03 与 T06 遇到服务器、延迟和反作弊报错时的双标签规则。
5. 长句是否直接多标签，还是需要进一步切成更短的分句。
6. 哪些短句只能进入 A01，不能作为玩法或剧情证据。

审查完成后，下一步会根据批准的种子计算主题原型，给 8,979 个句子生成多主题候选，再把结果汇总到 19,486 条评论。
