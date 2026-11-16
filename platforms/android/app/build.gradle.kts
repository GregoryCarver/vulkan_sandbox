import java.io.File

plugins {
    id("com.android.application")
}

val repositoryRoot = rootProject.projectDir.resolve("../..").canonicalFile
val slangCompiler = providers.gradleProperty("slangc").orElse(provider {
    val sdk = System.getenv("VULKAN_SDK")
    val windows = System.getProperty("os.name").startsWith("Windows")
    if (sdk != null) File(sdk, if (windows) "Bin/slangc.exe" else "bin/slangc").path
    else if (windows) "slangc.exe" else "slangc"
})

android {
    namespace = "dev.vulkansandbox.viewer"
    compileSdk = 35
    ndkVersion = "28.0.13004108"

    defaultConfig {
        applicationId = "dev.vulkansandbox.viewer"
        // Vulkan 1.3 loader symbols are exported starting at Android API 33.
        minSdk = 33
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DSANDBOX_BUILD_TESTS=OFF",
                    "-DSANDBOX_ENABLE_IMGUI=ON",
                    "-DSANDBOX_ANDROID_TARGET_FPS=${providers.gradleProperty("sandboxFps").getOrElse("60")}",
                    "-DSANDBOX_ANDROID_RENDER_SCALE=${providers.gradleProperty("sandboxRenderScale").getOrElse("1.0")}",
                    "-DSLANGC_EXECUTABLE=${slangCompiler.get()}"
                )
                targets += "sandbox_android"
            }
        }
    }
    buildFeatures { prefab = true }
    externalNativeBuild {
        cmake {
            path = repositoryRoot.resolve("CMakeLists.txt")
            version = "3.31.6"
        }
    }
    for (variant in listOf("debug", "release")) {
        sourceSets.getByName(variant).assets.srcDir(layout.buildDirectory.dir("generated/sandboxAssets/$variant"))
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildTypes {
        debug {
            isJniDebuggable = true
            externalNativeBuild.cmake.arguments +=
                "-DSANDBOX_ANDROID_ASSET_OUTPUT=${layout.buildDirectory.dir("generated/sandboxAssets/debug").get().asFile.path}"
        }
        release {
            isMinifyEnabled = false
            externalNativeBuild.cmake.arguments +=
                "-DSANDBOX_ANDROID_ASSET_OUTPUT=${layout.buildDirectory.dir("generated/sandboxAssets/release").get().asFile.path}"
            // Release signing belongs to the application using this template.
        }
    }
}

// CMake owns shader compilation. Package the exact same output as the desktop
// build; build it before AGP snapshots generated assets, including shader-only edits.
tasks.configureEach {
    if (name == "mergeDebugAssets") dependsOn("externalNativeBuildDebug")
    if (name == "mergeReleaseAssets") dependsOn("externalNativeBuildRelease")
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.games:games-activity:3.0.5")
}
