"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Code by: Evenset inc.
""" simple mask plugin for zipcode """

from masking_plugins.Mask_abstract import Mask_abstract
class mask_contact_simple(Mask_abstract):
    """Abstract class that other masking plugins should implement"""

    def mask(self, text_to_reduct):
        """Implementation of the method that should perform masking.
            Takes a token as input and returns a set string "CONTACT"
        """
        return "CONTACT"
