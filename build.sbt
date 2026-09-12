// whisper-si: Whisper-tiny accelerator generator (Chisel)
ThisBuild / scalaVersion := "2.13.18"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "whisper-si"

val chiselVersion = "7.15.0"

lazy val root = (project in file("."))
  .settings(
    name := "whisper-si",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.scalatest" %% "scalatest" % "3.2.19" % "test",
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations",
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
    Test / fork := true,
    Test / parallelExecution := false,
    Test / javaOptions ++= Seq("-Xmx8g", "-Xss64m"),
    // Weight hex files live in weights/ (source of truth); the RomInit backend reads them by absolute path.
    Compile / unmanagedResourceDirectories += baseDirectory.value / "weights",
  )
