![](./img/git.png)

- Remote：远程仓库
- Repository：本地仓库
- Index：暂存区
- Workspace：工作区

[git学习指南](https://blog.csdn.net/m0_46168595/article/details/114839387)

==这个讲的挺全面的，受益不浅，值得花心思看看。==

**简约版流程：**

1. git pull

   拉取代码

   若前面增加以下两条指令，可实现强制拉取代码覆盖本地代码。

   1. git fetch --all

      从远程仓库获取最新的代码和分支信息，但不会进行合并。

   2. git reset --hard origin/\<branch\>

      origin/branch：远程仓库的分支

       将会重置本地仓库，并使用远程仓库中指定分支的代码替换本地仓库中对应分支的代码。注意，该命令会删除本地未提交的所有改动和文件，慎重使用。 

      > branch 是你的本地代码分支名称
      >
      > 1和2一块使用，再使用3即可实现强制拉去远程代码，覆盖本地原有代码

2. git add .

   添加修改代码内容

3. git commit -m "..."

   提交，可写提交注释

4. git push

   推代码



**分支相关**

 ```git
# 列出所有本地分支
git branch

# 列出所有远程分支
git branch -r

# 列出所有本地分支和远程分支
git branch -a

# 新建一个分支，但依然停留在当前分支
git branch [branch-name]

# 以远程分支为基础新建一个分支，并切换到该分支
git checkout -b [branch] origin/[remote-branch]

# 新建一个分支，指向指定的commit
git branch [branch] [commit]

# 新建一个分支，与指定的远程分支建立追踪关系
git branch --track [branch] [remote-branch]

# 建立追踪关系，在现有分支与指定的远程分支之间
git branch --set-upstream [branch] [remote-branch]

# 切换到指定分支
git checkout [branch-name]

# 切换到上一个分支
git checkout -

# 合并指定分支到当前分支
git merge [branch]

# 选择一个commit，合并进当前分支
git cherry-pick [commit]
 
# 删除分支
git branch -d [branch-name]
 
# 删除远程分支
git push origin --delete [branch-name]
git branch -dr [remote/branch]
 ```





**本地分支关联远程：**

1. git branch --set-upstream-to=origin/分支名(本地) 分支名(远程)



**代码库修改密码后push不了？**

```git
// 重新输入密码
git config --system --unset credential.helper

// 密码存储同步
git config --global credential.helper store
```

