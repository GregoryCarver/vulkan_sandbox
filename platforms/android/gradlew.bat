@echo off
setlocal
set "APP_HOME=%~dp0"
if exist "%APP_HOME%gradle\wrapper\gradle-wrapper.jar" goto wrapper
where gradle >nul 2>nul
if errorlevel 1 (
    echo Gradle wrapper JAR has not been bootstrapped. Install Gradle 8.11.1 and JDK 17, 1>&2
    echo then run: gradle -p "%APP_HOME%" wrapper --gradle-version 8.11.1 --distribution-type bin 1>&2
    exit /b 1
)
call gradle --version | findstr /x /c:"Gradle 8.11.1" >nul
if errorlevel 1 (
    echo This project requires Gradle 8.11.1. Generate the pinned wrapper using the documented bootstrap command. 1>&2
    exit /b 1
)
call gradle -p "%APP_HOME%" %*
exit /b %ERRORLEVEL%
:wrapper
if defined JAVA_HOME (
    set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
) else (
    set "JAVA_EXE=java.exe"
)
"%JAVA_EXE%" -classpath "%APP_HOME%gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*
exit /b %ERRORLEVEL%
