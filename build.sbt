// minicpm-si: MiniCPM5-2B LLM accelerator generator (Chisel)
ThisBuild / scalaVersion := "2.13.18"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "minicpm-si"

val chiselVersion = "7.15.0"

lazy val root = (project in file("."))
  .settings(
    name := "minicpm-si",
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
    Test / javaOptions ++= Seq("-Xmx10g", "-Xss64m"),
  )
