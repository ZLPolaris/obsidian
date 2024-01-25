# 概念

## DevOps
https://zhuanlan.zhihu.com/p/141115367
在DevOps之前，从业人员使用**瀑布模型**或**敏捷开发模型**进行软件项目开发：瀑布模型或顺序模型是软件开发生命周期（SDLC）中的一种开创性方法,在这个模型中，软件开发成为一个线性过程，不同的阶段和任务被依次定位；而敏捷开发涉及各种方法的使用和SDLC中多个团队的协作。瀑布模型的线性和敏捷开发的跨功能性**无法确保快速、连续地交付无缺陷的软件应用程序。**

软件行业日益清晰地认识到:为了按时交付软件产品和服务，开发和运维工作必须紧密合作。这样的情况下，DevOps应运而生。

DevOps是一个简单的缩写词，源于“development（开发）”和“Operation（运维）”两个词，它涉及以特定的方式实践应用程序开发的任务。更广泛地说，软件开发和IT运维的结合被称为DevOps。
### 生命周期
持续发展、持续集成、持续测试、持续反馈、持续监测、持续部署、持续运维七个阶段。
**持续开发**
规划和软件编码，
**持续集成**




# 构建应用
## CI/CD
### 概念

Gitlab CI/CD 是一个内置在GitLab中的工具，用于通过持续方法进行软件开发：
Continuous Integration (CI)  持续集成
Continuous Delivery (CD)     持续交付
Continuous Deployment (CD)   持续部署
通过软件开发的持续方法，您可以持续构建、测试和部署迭代代码更改。这种迭代过程有助于减少您基于有缺陷或失败的先前版本开发新代码的机会。 使用这种方法，您可以努力减少从开发新代码到部署的人工干预，甚至根本不需要干预。
**持续集成**
考虑一个应用程序，它的代码存储在极狐GitLab 的 Git 存储库中。开发人员每天多次推送代码更改（甚至是开发分支）。对于每次推送到仓库，您可以创建一组脚本来自动构建和测试您的应用程序。这些脚本有助于减少您在应用程序中引入错误的机会。
**持续交付**
[持续交付](https://continuousdelivery.com/) 是超越持续集成的一步。每次将代码更改推送到代码库时，不仅会构建和测试您的应用程序，还会持续部署应用程序。但是，对于持续交付，您需要手动触发部署。
**持续部署**
[持续部署](https://www.airpair.com/continuous-deployment/posts/continuous-deployment-for-practical-people)是超越持续集成的又一步，类似于持续交付。不同之处在于，不是手动部署应用程序，而是将其设置为自动部署。不需要人工干预。
#### 工作流
您可以将提交推送到托管在极狐GitLab 中的远端仓库中的功能分支。 推送会触发项目的 CI/CD 流水线。然后，GitLab CI/CD：
- 运行自动化脚本（顺序或并行）：
    - 构建和测试您的应用程序。
    - 在 Review App 中预览更改，就像您在 `localhost` 上看到的一样。
实施后按预期工作：
- 审核并批准您的代码。
- 将功能分支合并到默认分支中。
    - GitLab CI/CD 将您的更改自动部署到生产环境。
![[gitlab_workflow_example_11_9.png]]
工作流程
![[gitlab_workflow_example_extended_v12_3.png]]



### 快速入门
条件：
确保有可用的runner
项目根目录创建一个.gitlab-ci.yml文件，用于定义CI/CD作业的地方
### Runner

### YAML文件
https://docs.gitlab.cn/jh/ci/yaml/#关键字
```yaml
# 关键字 配置流水线行为的全局关键字

# stages
# 定义包含作业组的阶段，在每个作业中使用stage定义作业属于那个阶段。
# 项的顺序就是作业的执行顺序，只有前面的项全部运行成功后才会运行。
# 默认的流水线阶段：.pre build test deploy .post
# 所有stage执行成功后流水线标记为passed，否则标记为failed
stages:
  - build
  - test
  - deploy
  - publish

# default 作业关键字的自定义默认值, 为某些关键字设置全局默认值。
default:
  image: ruby:3.0
  
# image镜像默认值为 ruby:3.0
rspec:
  script: bundle exec rspec
# 不使用默认值
rspec 2.7:
  image: ruby:2.7
  script: bundle exec rspec

# 控制运行的流水线类型。
workflow:
  name: 'Pipelinename'  # 字符串和CI/CD变量的混合
  
  # rules
  # 接受的关键字
  # if：检查此规则以确定何时运行流水线。
  #when：指定当 if 规则为 true 时要做什么。
  #要运行流水线，请设置为 always。
  #要阻止流水线运行，请设置为 never。
  #variables：如果未定义，则使用在别处定义的变量。
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
      when: never
    - if: '$CI_PIPELINE_SOURCE == "push"'
      when: never
    - when: always
  # 此示例阻止了计划或 push（分支和标签）的流水线。
  # 最后的 when: always 规则运行所有其他流水线类型，包括合并请求流水线。
  # 如果您的规则同时匹配分支流水线和合并请求流水线，则可能会出现重复流水线。
    - if: $CI_COMMIT_REF_NAME == $CI_DEFAULT_BRANCH
      variables:
        DEPLOY_VARIABLE: "deploy-production"  # Override globally-defined DEPLOY_VARIABLE
    - if: $CI_COMMIT_REF_NAME =~ /feature/
      variables:
        IS_A_FEATURE: "true"                  # Define a new variable.
    - when: always                            # Run the pipeline in other cases
  # 当条件匹配时，将创建该变量并可供流水线中的所有作业使用。
  # 如果该变量已在全局级别定义，则 workflow 变量优先并覆盖全局变量。
  
# include
# 从其他YAML文件导入配置
# 可能的输入：include 子键：
# include:local
# include:project
# include:remote
# include:template
include:
  - local: '/templates/.gitlab-ci-template.yml'

# 作业关键字
# image 指定运行作业的Docker镜像
# 可能的输入：镜像的名称，包括镜像库路径（如果需要），采用以下格式之一：
# <image-name>（与使用带有 latest 标签的 <image-name> 相同）
# <image-name>:<tag>
# <image-name>@<digest>
# 定义全局的作业镜像 未来可能废弃 建议在default中定义全局image
image:
  name: "registry.example.com/my/image:latest"

# variables 
# 为作业自定义变量 全局和作业中都可以使用
# 变量在 script、before_script 和 after_script 命令中始终可用以及某些关键字
variables:
  DEPLOY_ENVIRONMENT:
    value: "staging"
    # 描述信息
    description: "The deployment target. Change this variable to 'canary' or 'production' if needed."
deploy_job:
  stage: deploy
  script:
    - deploy-script --url $DEPLOY_SITE --path "/"
  environment: production


# 定义一个作业
build-job:
  stage: build  # 作业的阶段是build
  # 作业的脚本 可以执行代码，运行shell脚本。
  script:
    - echo "Hello, $GITLAB_USER_LOGIN!"
  image: 
    name: "registry.example.com/my/image:latest"

# before_script 
# 定义一系列命令，这些命令应该在每个作业的script前执行，但在artifacts恢复之后.
# 在作业和default关键字中使用，不建议全局使用
job:
  before_script:
    - echo "Execute this command before any 'script:' commands."
  script:
    - echo "This command executes after the job's 'before_script' commands."

# after_scipt
# 定义在每个作业之后运行的命令数组，包括失败的作业。
job1:
  script:
    - echo "An example script section."
  after_script:
    - echo "Execute this command after the `script` section completes."



test-job1:
  stage: test
  script:
    - echo "This job tests something"

test-job2:
  stage: test
  script:
    - echo "This job tests something, but takes more time than test-job1."
    - echo "After the echo commands complete, it runs the sleep command for 20 seconds"
    - echo "which simulates a test that runs 20 seconds longer than test-job1"
    - sleep 20

deploy-prod:
  stage: deploy
  script:
    - echo "This job deploys something from the $CI_COMMIT_BRANCH branch."
  environment: production
```
### 作业
流水线配置从作业开始。作业是 `.gitlab-ci.yml` 文件中最基本的元素。
要求：
- 定义了约束条件，说明它们应该在什么条件下执行。
- 具有任意名称的顶级元素，并且必须至少包含 [`script`](https://docs.gitlab.cn/jh/ci/yaml/index.html#script) 子句。
- 不限制可以定义的数量。
### 流水线
流水线是持续集成、交付和部署的顶级组件。
流水线包括：
- 工作，定义做什么。例如，编译或测试代码的作业。
- 阶段，定义何时运行作业。例如，在编译代码的阶段之后运行测试的阶段。
#### 流水线类型
基本流水线：同时运行每个阶段的所有内容，然后就是下一个阶段。
有向无环图流水线：基于作业之间的关系。
多项目流水线：
父子流水线：
合并请求的流水线：
合并结果的流水线：
合并队列：



