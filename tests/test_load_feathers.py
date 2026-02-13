#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `enterprise_extensions` package."""

import os
from collections import Counter
import pytest

from enterprise_extensions.load_feathers import load_feathers_from_folder
from enterprise.pulsar import FeatherPulsar

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')
outdir = os.path.join(testdir, 'test_out')


class TestLoadFeathersFromFolder:
    DATA_DIR = datadir

    @pytest.fixture(scope="class")
    def pulsar_names(self, feather_filenames):
        """Extract pulsar names from feather filenames."""
        return [f.split('_')[0] for f in feather_filenames]

    @pytest.fixture(scope="class")
    def data_directory(self):
        """Fixture to check if the data directory exists before tests run."""
        if not os.path.isdir(self.DATA_DIR):
            pytest.skip(f"Data directory {self.DATA_DIR} not found, skipping tests.")
        return self.DATA_DIR

    @pytest.fixture(scope="class")
    def feather_filenames(self, data_directory):
        """Get list of feather filenames in the data directory."""
        feather_files = [f for f in os.listdir(data_directory) if f.endswith('.feather')]
        if not feather_files:
            pytest.skip(f"No feather files found in {data_directory}, skipping tests.")
        return feather_files

    def test_load_all_pulsars(self, data_directory, feather_filenames):
        """Test loading all pulsar files from the real directory."""
        # Load all pulsars
        pulsars = load_feathers_from_folder(data_directory)

        # Verify that we got the expected number of pulsars
        # Note: This assumes all .feather files pass any internal filters
        # If your function has internal filters, the actual count might be lower
        assert len(pulsars) <= len(feather_filenames)

        # Verify that each pulsar is a Pulsar object
        for psr in pulsars:
            assert isinstance(psr, FeatherPulsar)

    def test_filter_by_pulsar_name(self, data_directory, pulsar_names):
        """Test filtering by pulsar name with real data."""
        if not pulsar_names:
            pytest.skip("No pulsar names available for testing.")

        # Select the first pulsar name for testing
        test_pulsar = pulsar_names[0]

        # Count the number of times this pulsar appears in the list
        pulsar_count = Counter(pulsar_names)[test_pulsar]

        # Load only that specific pulsar
        pulsars = load_feathers_from_folder(data_directory, pulsar_name_list=[test_pulsar])

        # Check that we got at least one pulsar back
        assert len(pulsars) == pulsar_count

        # If we got a pulsar, make sure it has the correct name
        if pulsars:
            assert pulsars[0].name == test_pulsar

    def test_exclude_pattern(self, data_directory, pulsar_names):
        """Test excluding pulsars by pattern with real data."""
        if not pulsar_names:
            pytest.skip("No pulsar names available for testing.")

        # Choose a pattern that appears in at least one filename
        pattern_to_exclude = pulsar_names[0]

        # Load pulsars excluding the chosen pattern
        all_pulsars = load_feathers_from_folder(data_directory)
        filtered_pulsars = load_feathers_from_folder(data_directory, exclude_pattern=pattern_to_exclude)

        # Verify fewer pulsars when using exclude_pattern
        assert len(filtered_pulsars) <= len(all_pulsars)

        # Verify none of the returned pulsars have names containing the excluded pattern
        for psr in filtered_pulsars:
            assert pattern_to_exclude not in psr.name

    def test_time_span_filter(self, data_directory):
        """Test filtering by time span with real data."""
        # First load all pulsars to see what time spans we're working with
        all_pulsars = load_feathers_from_folder(data_directory)

        if not all_pulsars:
            pytest.skip("No pulsars loaded, skipping time span test.")

        # Calculate the time spans for all pulsars
        time_spans = []
        for psr in all_pulsars:
            span_years = (psr.toas.max() - psr.toas.min()) / (525600 * 60)
            time_spans.append((psr.name, span_years))

        if not time_spans:
            pytest.skip("No time spans calculated, skipping test.")

        # Find the median time span
        time_spans.sort(key=lambda x: x[1])
        median_span = time_spans[len(time_spans) // 2][1]

        # Use the median as the cutoff to ensure we get a mix of results
        with pytest.MonkeyPatch().context() as mp:
            # Temporarily redirect print to avoid cluttering the test output
            mp.setattr('builtins.print', lambda *args, **kwargs: None)

            filtered_pulsars = load_feathers_from_folder(data_directory, time_span_cut_yr=median_span)

        # Check that we got fewer pulsars than before
        assert len(filtered_pulsars) <= len(all_pulsars)

        # Verify all returned pulsars have time spans >= the cutoff
        for psr in filtered_pulsars:
            span_years = (psr.toas.max() - psr.toas.min()) / (525600 * 60)
            assert span_years >= median_span
