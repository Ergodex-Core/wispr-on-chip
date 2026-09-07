package whisper

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

class SmokeSpec extends AnyFlatSpec {
  "Smoke" should "add on verilator via svsim" in {
    simulate(new Smoke) { dut =>
      dut.io.a.poke(3.U)
      dut.io.b.poke(4.U)
      dut.clock.step()
      dut.io.y.expect(7.U)
    }
  }
}
