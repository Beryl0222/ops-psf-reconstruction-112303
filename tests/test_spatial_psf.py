import unittest

import numpy as np

import psf_reconstruction as psf
from psf_reconstruction_input import load_record


def tiny_problem():
    basis = np.zeros((2, 3, 3))
    basis[0, 1, 1] = 1.0
    basis[1, 1, 1] = 0.5
    basis[1, 1, 2] = 0.5
    calibrations = {"slow": np.array([[[1.0, 0.0]]]), "fast": np.array([[[0.0, 1.0]]])}
    frame = psf.Frame(
        id="f",
        revision=1,
        calibration="fast",
        origin=(1, 1),
        pixels=np.array([[1.0, 2.0], [3.0, 4.0]]),
        invalid=[(0, 1)],
        noise_variance=0.25,
    )
    op = psf.SpatialPSFOperator(
        frames=[frame],
        part_shape=(4, 5),
        kernel_basis=basis,
        calibrations=calibrations,
        knots_y=[1],
        knots_x=[2],
    )
    return op, frame


class OperatorContractTests(unittest.TestCase):
    def test_raw_masked_weighted_and_whitened_adjoints(self):
        op, frame = tiny_problem()
        rng = np.random.default_rng(4)
        image = rng.normal(size=op.part_shape)
        upstream = rng.normal(size=frame.shape)

        pred = op.predict_frame(frame, image)
        self.assertLess(
            abs(np.vdot(pred, upstream) - np.vdot(image, op.adjoint_frame(frame, upstream))),
            1e-12,
        )

        residual = rng.normal(size=frame.shape)
        lhs = op.weighted_inner_product([pred], [residual])
        rhs = np.vdot(image, op.weighted_adjoint([residual]))
        self.assertLess(abs(lhs - rhs), 1e-12)

        xvec = rng.normal(size=op.part_shape)
        yvec = rng.normal(size=op.n_observations)
        self.assertLess(
            abs(np.dot(op.observed_forward(xvec), yvec) - np.vdot(xvec, op.observed_adjoint(yvec))),
            1e-12,
        )
        zvec = rng.normal(size=op.n_observations)
        self.assertLess(
            abs(np.dot(op.whitened_forward(xvec), zvec) - np.vdot(xvec, op.whitened_adjoint(zvec))),
            1e-12,
        )

    def test_zero_valid_pixel_is_still_observation_and_invalid_is_excluded(self):
        record = load_record()
        op = psf.build_operator(record, which="initial")
        frame = op.frames[0]
        # over-a contains valid zero samples at local (0,0), (0,1), and (2,2).
        self.assertEqual(frame.pixels[2, 2], 0)
        self.assertTrue(frame.valid_mask[2, 2])
        self.assertFalse(frame.valid_mask[1, 2])
        obs = op.observability()
        self.assertTrue(obs.direct_observed[np.nonzero(obs.direct_observed)][0])
        self.assertFalse(obs.direct_observed[0, 0])

    def test_tiled_matches_single_tile(self):
        record = load_record()
        frame = psf.Frame.from_mapping(record["frames"][0])
        cal = psf.CalibrationSet.from_record(record)
        big = psf.SpatialPSFOperator(
            frames=[frame], calibration=cal, part_shape=record["part_shape"],
            memory=psf.MemoryPolicy(tile_rows=1),
        )
        one = psf.SpatialPSFOperator(
            frames=[frame], calibration=cal, part_shape=record["part_shape"],
            memory=psf.MemoryPolicy(tile_rows=99),
        )
        x = np.random.default_rng(0).normal(size=record["part_shape"])
        u = np.random.default_rng(1).normal(size=frame.shape)
        np.testing.assert_allclose(big.predict_frame(frame, x), one.predict_frame(frame, x))
        np.testing.assert_allclose(big.adjoint_frame(frame, u), one.adjoint_frame(frame, u))


class ReconstructionTests(unittest.TestCase):
    def test_minimum_norm_underdetermined_solution(self):
        basis = np.zeros((1, 1, 1))
        basis[0, 0, 0] = 1.0
        op = psf.SpatialPSFOperator(
            frames=[psf.Frame(id="f", calibration="a", pixels=np.array([[3.0]]))],
            part_shape=(2, 2),
            kernel_basis=basis,
            calibrations={"a": np.ones((1, 1, 1))},
            knots_y=[0],
            knots_x=[0],
        )
        result = op.reconstruct(smoothness=0.0, tolerance=1e-12)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.image, [[3, 0], [0, 0]], atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(result.x), 3.0)

    def test_unconverged_candidate_is_flagged(self):
        op, _ = tiny_problem()
        result = op.reconstruct(smoothness=0.01, tolerance=1e-30, max_iterations=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "max_iterations")
        self.assertEqual(result.iterations, 0)
        self.assertTrue(hasattr(result, "image"))

    def test_smoothed_unobserved_pixels_are_marked_missing(self):
        record = load_record()
        result = psf.reconstruct(
            psf.build_operator(record, which="initial"),
            smoothness=record["smoothness"],
            tolerance=1e-10,
        )
        self.assertTrue(result.converged)
        window = result.inclusive_window(record["requested_roi"])
        self.assertTrue(np.any(window.missing_direct_observation))
        # Smoothing nevertheless supplies finite candidate values there.
        missing = window.missing_direct_observation
        self.assertTrue(np.all(np.isfinite(window.image[missing])))
        self.assertGreater(len(window.missing_coordinates), 0)

    def test_domain_edges_only(self):
        domain = np.array([[True, True, False], [True, False, True]])
        x = np.arange(6, dtype=float).reshape(2, 3)
        lap = psf.laplacian_matvec(x, domain, 1.0)
        # Edge (0,0)-(1,0) exists; (0,1) has no edge to masked (1,1).
        self.assertNotEqual(lap[0, 0], 0)
        self.assertEqual(lap[0, 2], 0)


class ReplacementAndGradientTests(unittest.TestCase):
    def test_same_revision_conflict_and_higher_revision_replaces_support(self):
        record = load_record()
        session = psf.ScanSession.from_record(record)
        old_b = session.registry.get("over-b", 1)
        new_b = session.registry.get("over-b", 2)
        self.assertEqual(old_b.calibration, "fast")
        self.assertEqual(new_b.calibration, "slow")

        conflict = old_b.copy()
        conflict.pixels = conflict.pixels + 1
        with self.assertRaises(psf.FrameConflictError):
            session.replace_frame(conflict)
        stale = psf.Frame.from_mapping(record["frames"][0])
        stale.id = "over-b"
        stale.revision = 1
        with self.assertRaises(psf.RevisionConflictError):
            session.replace_frame(stale)

        current_refs = session.operator.refs
        self.assertIn(("over-b", 2), current_refs)
        self.assertNotIn(("over-b", 1), current_refs)
        self.assertEqual(session.frame_set("initial").n_observations, 16)
        self.assertEqual(session.operator.n_observations, 15)
        with self.assertRaises(KeyError):
            session.operator._frame_index_for_key(("over-b", 1))

    def test_locked_vs_current_comparison(self):
        workflow = psf.run_record_workflow()
        comp = workflow["comparison"]
        self.assertIn(("over-b", 1), workflow["locked_result"].frame_refs)
        self.assertIn(("over-b", 2), workflow["current_result"].frame_refs)
        self.assertEqual(comp.difference.shape, workflow["locked_result"].shape)
        self.assertGreater(comp.metrics["l2_norm"], 0)

    def test_gradient_uses_original_fast_call_and_matches_fd(self):
        workflow = psf.run_record_workflow()
        gradient = workflow["gradient"]
        self.assertEqual(gradient.calibration_name, "fast")
        self.assertEqual(gradient.frame_ref, ("over-b", 1))

        record = workflow["record"]
        cal = psf.CalibrationSet.from_record(record)
        frame = psf.Frame.from_mapping(record["frames"][1])
        image = workflow["locked_result"].image
        upstream = np.asarray(record["gradient_probe"]["upstream"], float)
        expected = psf.calibration_parameter_gradient(cal, frame, image, upstream)

        base = cal.profiles["fast"].copy()
        fd = np.zeros_like(base)
        eps = 1e-6
        for index in np.ndindex(base.shape):
            cal.profiles["fast"] = base.copy()
            cal.profiles["fast"][index] += eps
            plus = psf.SpatialPSFOperator(
                frames=[frame], calibration=cal, part_shape=record["part_shape"]
            ).predict_frame(frame, image)
            cal.profiles["fast"] = base.copy()
            cal.profiles["fast"][index] -= eps
            minus = psf.SpatialPSFOperator(
                frames=[frame], calibration=cal, part_shape=record["part_shape"]
            ).predict_frame(frame, image)
            fd[index] = np.sum(upstream * (plus - minus)) / (2 * eps)
        cal.profiles["fast"] = base
        np.testing.assert_allclose(expected, fd, atol=2e-6)

        with self.assertRaises(psf.CalibrationMismatchError):
            workflow["session"].calibration_gradient(
                "gradient-probe-original-call", upstream, expected_calibration="slow"
            )


if __name__ == "__main__":
    unittest.main()
